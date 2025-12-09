"""
Gaussian-aware view selection strategy.

This selector prioritizes cameras based on which Gaussians they can see,
enabling adaptive sampling that focuses on under-trained or sparse regions.
"""

import numpy as np
import torch
from typing import Dict, List
from scipy.special import softmax
from .selector import ViewSelector


class GaussianAwareSelector(ViewSelector):
    """
    Selector that adapts to the current Gaussian distribution.

    Two modes:
    - 'inverse_density': Prioritize cameras seeing fewer Gaussians (sparse regions)
    - 'coverage_gap': Prioritize cameras seeing under-trained Gaussians

    Uses frustum culling to determine which Gaussians are visible to each camera.
    """

    def __init__(self, config: dict = None, log_dir: str = None, verbose: bool = False, seed: int = None):
        """
        Initialize the Gaussian-aware selector.

        Args:
            config: Configuration dictionary with optional keys:
                - mode (str): 'inverse_density' or 'coverage_gap'. Default: 'inverse_density'
                - update_frequency (int): How often to recompute Gaussian stats (iterations).
                  Default: 500 (expensive operation)
                - temperature (float): Softmax temperature. Default: 1.0
                - frustum_margin (float): Margin for frustum check (1.0 = no margin). Default: 1.2
                - near_plane (float): Near clipping plane distance. Default: 0.01
                - far_plane (float): Far clipping plane distance. Default: 100.0
            log_dir: Directory to save selection logs
            verbose: If True, print detailed information
            seed: Random seed for reproducibility
        """
        super().__init__(config, log_dir, verbose, seed)

        self.mode = self.config.get('mode', 'inverse_density')
        self.update_frequency = self.config.get('update_frequency', 500)
        self.temperature = self.config.get('temperature', 1.0)
        self.frustum_margin = self.config.get('frustum_margin', 1.2)
        self.near_plane = self.config.get('near_plane', 0.01)
        self.far_plane = self.config.get('far_plane', 100.0)

        # Cached data
        self.camera_gaussian_counts = {}  # Maps camera uid to number of visible Gaussians
        self.coverage_counts = None  # Tensor tracking training count per Gaussian
        self.last_update_iteration = -self.update_frequency  # Force initial update
        self.current_probabilities = None

        # Pending coverage updates (for lazy batch processing)
        self.pending_coverage_cameras = []  # List of cameras waiting for coverage update

        if self.verbose:
            print(f"[GaussianAwareSelector] Configuration:")
            print(f"  Mode: {self.mode}")
            print(f"  Update frequency: {self.update_frequency} iterations")
            print(f"  Temperature: {self.temperature}")
            print(f"  Frustum margin: {self.frustum_margin}")

    def initialize(self, all_cameras: List) -> None:
        """
        Initialize camera list and coverage tracking.

        Args:
            all_cameras: List of all available Camera objects
        """
        super().initialize(all_cameras)

        # Initialize camera-Gaussian counts (will be computed on first call)
        self.camera_gaussian_counts = {cam.uid: 0 for cam in all_cameras}

        self.initialized = True

        if self.verbose:
            print(f"[GaussianAwareSelector] Initialized with {len(all_cameras)} cameras")

    def _check_frustum_batch(self, gaussian_positions: torch.Tensor, camera) -> torch.Tensor:
        """
        Check which Gaussians are visible from a camera (vectorized).

        Args:
            gaussian_positions: Tensor of shape [N, 3] with Gaussian positions
            camera: Camera object

        Returns:
            Boolean tensor of shape [N] indicating visibility
        """
        # Transform Gaussians to camera space
        # camera.world_view_transform is 4x4 matrix (world to camera)
        n_gaussians = gaussian_positions.shape[0]

        # Add homogeneous coordinate
        ones = torch.ones((n_gaussians, 1), device=gaussian_positions.device)
        positions_homo = torch.cat([gaussian_positions, ones], dim=1)  # [N, 4]

        # Transform to camera space
        # world_view_transform is already transposed in the Camera class
        cam_transform = camera.world_view_transform.to(gaussian_positions.device)
        positions_cam = (cam_transform @ positions_homo.T).T  # [N, 4]

        # Extract camera-space coordinates
        x_cam = positions_cam[:, 0]
        y_cam = positions_cam[:, 1]
        z_cam = positions_cam[:, 2]
        w_cam = positions_cam[:, 3]

        # Depth check: Gaussian should be in front of camera
        depth_valid = (z_cam > self.near_plane) & (z_cam < self.far_plane)

        # Project to normalized device coordinates (NDC)
        # Simple perspective projection: x_ndc = x_cam / z_cam, y_ndc = y_cam / z_cam
        # We use a margin to be more inclusive
        margin = self.frustum_margin

        # Avoid division by zero
        z_safe = torch.where(z_cam > 1e-6, z_cam, torch.ones_like(z_cam))

        x_ndc = x_cam / z_safe
        y_ndc = y_cam / z_safe

        # Compute frustum bounds based on actual camera FOV (if available)
        # Use camera.FoVx and camera.FoVy for more accurate frustum check
        if hasattr(camera, 'FoVx') and hasattr(camera, 'FoVy'):
            # Compute bounds from FOV
            # tan(FOV/2) gives the half-width/half-height at distance 1
            fov_x_half = np.tan(camera.FoVx / 2.0)
            fov_y_half = np.tan(camera.FoVy / 2.0)

            # Apply margin
            bound_x = fov_x_half * margin
            bound_y = fov_y_half * margin

            frustum_valid = (
                (x_ndc >= -bound_x) & (x_ndc <= bound_x) &
                (y_ndc >= -bound_y) & (y_ndc <= bound_y)
            )
        else:
            # Fallback: use fixed bound
            bound = margin
            frustum_valid = (
                (x_ndc >= -bound) & (x_ndc <= bound) &
                (y_ndc >= -bound) & (y_ndc <= bound)
            )

        # Combine checks
        visible = depth_valid & frustum_valid

        return visible

    def _compute_gaussian_visibility(self, gaussians) -> None:
        """
        Compute how many Gaussians each camera can see.

        Args:
            gaussians: Current Gaussian model with get_xyz method
        """
        # Get current Gaussian positions
        gaussian_positions = gaussians.get_xyz.detach()  # [N, 3]
        n_gaussians = gaussian_positions.shape[0]

        # Initialize coverage counts if needed (for coverage_gap mode)
        if self.mode == 'coverage_gap' and self.coverage_counts is None:
            self.coverage_counts = torch.zeros(n_gaussians, device=gaussian_positions.device)

        # EDGE CASE: Reset coverage counts if Gaussian count changed (densification/pruning)
        if self.coverage_counts is not None and self.coverage_counts.shape[0] != n_gaussians:
            if self.verbose:
                print(f"[GaussianAwareSelector] Gaussian count changed: "
                      f"{self.coverage_counts.shape[0]} -> {n_gaussians}. Resetting coverage counts.")
            self.coverage_counts = torch.zeros(n_gaussians, device=gaussian_positions.device)

        # Compute visibility for each camera
        for cam in self.all_cameras:
            visible = self._check_frustum_batch(gaussian_positions, cam)
            visible_count = visible.sum().item()
            self.camera_gaussian_counts[cam.uid] = visible_count

        if self.verbose:
            counts = list(self.camera_gaussian_counts.values())
            print(f"[GaussianAwareSelector] Gaussian visibility computed:")
            print(f"  Total Gaussians: {n_gaussians}")
            print(f"  Visible range: [{min(counts)}, {max(counts)}]")
            print(f"  Mean visible: {np.mean(counts):.1f}")

    def _compute_probabilities_internal(self, gaussians) -> Dict[int, float]:
        """
        Compute probabilities based on Gaussian visibility.

        Args:
            gaussians: Current Gaussian model

        Returns:
            Dictionary mapping camera uid to probability
        """
        if self.mode == 'inverse_density':
            # Prioritize cameras seeing fewer Gaussians (sparse regions)
            scores = []
            uids = []

            max_count = max(self.camera_gaussian_counts.values()) if self.camera_gaussian_counts else 1.0

            for cam in self.all_cameras:
                count = self.camera_gaussian_counts[cam.uid]
                # Inverse score: fewer Gaussians = higher score
                # Add 1 to avoid division by zero
                score = 1.0 / (count + 1.0)
                # Normalize by max count to keep scores reasonable
                score = score * max_count
                scores.append(score)
                uids.append(cam.uid)

        elif self.mode == 'coverage_gap':
            # Prioritize cameras seeing under-trained Gaussians
            gaussian_positions = gaussians.get_xyz.detach()

            scores = []
            uids = []

            for cam in self.all_cameras:
                visible = self._check_frustum_batch(gaussian_positions, cam)

                if visible.sum() > 0:
                    # Average coverage count of visible Gaussians
                    # Lower average = more under-trained Gaussians
                    visible_coverage = self.coverage_counts[visible]
                    avg_coverage = visible_coverage.mean().item()

                    # Inverse: lower coverage = higher score
                    score = 1.0 / (avg_coverage + 1.0)
                else:
                    # No visible Gaussians, give low score
                    score = 0.1

                scores.append(score)
                uids.append(cam.uid)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Convert to probabilities using softmax
        scores = np.array(scores)
        probabilities = softmax(scores / self.temperature)

        probabilities_dict = {
            uid: float(probabilities[i])
            for i, uid in enumerate(uids)
        }

        return probabilities_dict

    def compute_probabilities(self, gaussians, iteration: int) -> Dict[int, float]:
        """
        Compute sampling probabilities based on Gaussian visibility.

        Updates Gaussian visibility stats every update_frequency iterations.

        Args:
            gaussians: Current Gaussian model
            iteration: Current training iteration

        Returns:
            Dictionary mapping camera uid to probability
        """
        # Check if we need to recompute Gaussian visibility
        if iteration - self.last_update_iteration >= self.update_frequency:
            if self.verbose:
                print(f"\n[GaussianAwareSelector] Updating Gaussian visibility at iteration {iteration}")

            self._compute_gaussian_visibility(gaussians)
            self.current_probabilities = self._compute_probabilities_internal(gaussians)
            self.last_update_iteration = iteration

        # If probabilities haven't been computed yet (first call)
        if self.current_probabilities is None:
            self._compute_gaussian_visibility(gaussians)
            self.current_probabilities = self._compute_probabilities_internal(gaussians)
            self.last_update_iteration = iteration

        return self.current_probabilities

    def log_selection(self, cam, score: float, iteration: int) -> None:
        """
        Record selection and update coverage counts if in coverage_gap mode.

        Args:
            cam: Selected Camera object
            score: Probability that led to this selection
            iteration: Current training iteration
        """
        # Call parent logging
        super().log_selection(cam, score, iteration)

        # BUGFIX: Track cameras for coverage update in coverage_gap mode
        if self.mode == 'coverage_gap' and self.coverage_counts is not None:
            # Add camera to pending list for batch processing
            self.pending_coverage_cameras.append(cam)

            if self.verbose and len(self.pending_coverage_cameras) % 100 == 0:
                print(f"[GaussianAwareSelector] {len(self.pending_coverage_cameras)} cameras pending coverage update")

    def update_coverage_counts(self, gaussians, camera=None) -> None:
        """
        Update coverage counts for Gaussians visible to cameras.

        This should be called periodically from the training loop.
        If no camera is provided, processes all pending cameras from the queue.

        Args:
            gaussians: Current Gaussian model
            camera: Optional camera to update. If None, processes pending queue.
        """
        if self.mode != 'coverage_gap' or self.coverage_counts is None:
            return

        gaussian_positions = gaussians.get_xyz.detach()

        # Process single camera or pending queue
        cameras_to_process = [camera] if camera is not None else self.pending_coverage_cameras

        if not cameras_to_process:
            return

        total_updated = 0
        for cam in cameras_to_process:
            visible = self._check_frustum_batch(gaussian_positions, cam)
            # Increment counts for visible Gaussians
            self.coverage_counts[visible] += 1
            total_updated += visible.sum().item()

        # Clear pending queue if we processed it
        if camera is None:
            self.pending_coverage_cameras = []

        if self.verbose and total_updated > 0:
            print(f"[GaussianAwareSelector] Updated coverage for {total_updated} Gaussian views "
                  f"({len(cameras_to_process)} cameras)")

    def get_gaussian_statistics(self) -> Dict:
        """
        Get statistics about Gaussian visibility and coverage.

        Returns:
            Dictionary with Gaussian-related statistics
        """
        stats = {
            'mode': self.mode,
            'camera_gaussian_counts': self.camera_gaussian_counts.copy(),
        }

        if self.camera_gaussian_counts:
            counts = list(self.camera_gaussian_counts.values())
            stats['visibility_stats'] = {
                'min_visible': int(min(counts)),
                'max_visible': int(max(counts)),
                'mean_visible': float(np.mean(counts)),
                'std_visible': float(np.std(counts)),
            }

        if self.mode == 'coverage_gap' and self.coverage_counts is not None:
            stats['coverage_stats'] = {
                'min_coverage': float(self.coverage_counts.min().item()),
                'max_coverage': float(self.coverage_counts.max().item()),
                'mean_coverage': float(self.coverage_counts.mean().item()),
                'std_coverage': float(self.coverage_counts.std().item()),
                'total_gaussians': int(self.coverage_counts.shape[0]),
            }

        return stats

    def log_statistics(self, iteration: int) -> None:
        """
        Print Gaussian visibility and selection statistics.

        Args:
            iteration: Current training iteration
        """
        # Get base selection statistics
        selection_stats = self.get_selection_statistics()

        # Get Gaussian statistics
        gaussian_stats = self.get_gaussian_statistics()

        print(f"\n[GaussianAwareSelector] Statistics at iteration {iteration}:")
        print(f"  Mode: {self.mode}")
        print(f"  Selection stats:")
        print(f"    Total selections: {selection_stats.get('total_selections', 0)}")
        print(f"    Unique cameras: {selection_stats.get('unique_cameras', 0)}")

        if 'visibility_stats' in gaussian_stats:
            vstats = gaussian_stats['visibility_stats']
            print(f"  Gaussian visibility:")
            print(f"    Range: [{vstats['min_visible']}, {vstats['max_visible']}]")
            print(f"    Mean ± std: {vstats['mean_visible']:.1f} ± {vstats['std_visible']:.1f}")

        if 'coverage_stats' in gaussian_stats:
            cstats = gaussian_stats['coverage_stats']
            print(f"  Gaussian coverage:")
            print(f"    Range: [{cstats['min_coverage']:.1f}, {cstats['max_coverage']:.1f}]")
            print(f"    Mean ± std: {cstats['mean_coverage']:.1f} ± {cstats['std_coverage']:.1f}")
