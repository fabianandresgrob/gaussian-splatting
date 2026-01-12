"""
Geometric diversity view selection based on pose heuristics.

This selector computes probabilities based on camera pose properties (geometric diversity).
Supports both static and dynamic modes:
- Static: Probabilities computed once at initialization (cheap but doesn't adapt)
- Dynamic: Probabilities updated based on recently selected views (adapts to avoid redundancy)
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from scipy.special import softmax
from .selector import ViewSelector


class GeometricDiversitySelector(ViewSelector):
    """
    Selector that computes probabilities based on geometric diversity heuristics.

    Uses camera poses to identify "unique" viewpoints through:
    - Spatial diversity: Average distance to k nearest neighbors (static mode)
    - Distance to recently selected cameras (dynamic mode)

    Both modes can optionally include viewing direction in distance calculations.

    Modes:
    - 'static': Probabilities computed once at init based on k-NN diversity
    - 'distance_to_selected': Dynamic mode that prioritizes cameras far from recent selections
    """

    def __init__(self, config: dict = None, log_dir: str = None, verbose: bool = False, seed: int = None):
        """
        Initialize the geometric diversity selector.

        Args:
            config: Configuration dictionary with optional keys:
                - mode (str): 'static' or 'distance_to_selected'. Default: 'distance_to_selected'
                - temperature (float): Softmax temperature for probability distribution.
                  Higher values make distribution more uniform. Default: 1.0
                - use_orientation (bool): Include viewing direction in distance calculations. Default: True
                - k_neighbors (int): Number of neighbors for k-NN diversity (static mode). Default: 5
                - recency_window (int): Number of recent selections to consider (dynamic mode). Default: 69
                - use_cumulative_penalty (bool): Apply penalty based on total selection count. Default: False
            log_dir: Directory to save selection logs
            verbose: If True, print detailed information
            seed: Random seed for reproducibility
        """
        super().__init__(config or {}, log_dir, verbose, seed=seed)
        self.mode = self.config.get('mode', 'distance_to_selected')
        self.temperature = self.config.get('temperature', 1.0)
        self.use_orientation = self.config.get('use_orientation', True)
        self.k_neighbors = self.config.get('k_neighbors', 5)
        self.recency_window = self.config.get('recency_window', 69)
        # NOTE: Disabled by default - this tracks by camera uid, not by pose similarity,
        # so duplicates with different uids would be tracked separately. The recency-based
        # approach using pose distance is the fair mechanism for diversity.
        self.use_cumulative_penalty = self.config.get('use_cumulative_penalty', False)
        
        # Static mode storage
        self.selection_probabilities = None
        
        # Dynamic mode storage
        self.pose_features = None  # (N, D) array of pose features (position + optional orientation)
        self.positions = None  # (N, 3) array of camera positions (kept for compatibility)
        self.recent_selections = []  # List of recently selected camera indices
        self.selection_counts = None  # Cumulative selection counts per camera
        self.cam_uid_to_idx = {}  # Map camera uid to index

    def initialize(self, all_cameras: List) -> None:
        """
        Initialize based on camera poses.

        Args:
            all_cameras: List of all available Camera objects
        """
        super().initialize(all_cameras)
        self.logger.info(f"Initializing with {len(all_cameras)} cameras, mode={self.mode}")
        self.logger.debug(f"Temperature: {self.temperature}, use_orientation: {self.use_orientation}")
        
        # Extract pose features (position + optional viewing direction)
        features = []
        positions = []
        for cam in all_cameras:
            pos = cam.camera_center.cpu().numpy()
            positions.append(pos)
            feature = list(pos)
            
            if self.use_orientation:
                # Compute viewing direction from rotation matrix
                # The camera looks down the negative Z axis in camera space
                # Transform to world space using R^T (since R transforms world to camera)
                R = cam.R  # world-to-camera rotation
                view_dir = -R[2, :]  # Third row gives the forward direction
                feature.extend(view_dir)
            
            features.append(feature)
        
        self.positions = np.array(positions)
        self.pose_features = np.array(features)
        
        # Normalize features for fair distance computation
        # Position and orientation may have different scales
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.pose_features_scaled = self.scaler.fit_transform(self.pose_features)
        
        # Build uid to index mapping
        self.cam_uid_to_idx = {cam.uid: i for i, cam in enumerate(all_cameras)}
        
        # Initialize selection tracking
        self.selection_counts = np.zeros(len(all_cameras))
        self.recent_selections = []

        if self.mode == 'static':
            self._compute_static_probabilities()
        else:
            self.logger.info(f"Dynamic mode: recency_window={self.recency_window}, "
                           f"cumulative_penalty={self.use_cumulative_penalty}")

        self.initialized = True

    def _compute_static_probabilities(self) -> None:
        """Compute fixed probabilities based on k-NN spatial diversity.
        
        Cameras that are isolated (far from their nearest neighbors in pose space)
        get higher scores, as they represent unique viewpoints.
        """
        n_cameras = len(self.all_cameras)
        k = min(self.k_neighbors, n_cameras - 1)
        
        self.logger.debug(f"Computing k-NN diversity with k={k}")
        
        # Spatial diversity: average distance to k nearest neighbors in pose space
        # Uses scaled features so position and orientation contribute fairly
        diversity_scores = np.zeros(n_cameras)
        
        for i in range(n_cameras):
            # Compute distances in scaled pose space
            dists = np.linalg.norm(
                self.pose_features_scaled - self.pose_features_scaled[i], 
                axis=1
            )
            # Get k nearest (excluding self at index 0 after sorting)
            nearest_dists = np.sort(dists)[1:k+1]
            diversity_scores[i] = np.mean(nearest_dists) if len(nearest_dists) > 0 else 0.0
        
        # Normalize to [0, 1]
        if diversity_scores.max() > diversity_scores.min():
            diversity_scores = (diversity_scores - diversity_scores.min()) / \
                              (diversity_scores.max() - diversity_scores.min())
        else:
            diversity_scores = np.ones(n_cameras)

        # Convert to probabilities
        probabilities = softmax(diversity_scores / self.temperature)

        self.selection_probabilities = {
            cam.uid: float(probabilities[i])
            for i, cam in enumerate(self.all_cameras)
        }

        self.logger.info(f"Probability range: {min(probabilities):.4f} - {max(probabilities):.4f}")

    def _compute_distance_to_selected_scores(self) -> np.ndarray:
        """
        Compute scores based on distance to recently selected cameras in pose space.
        
        Cameras far from recent selections (considering both position and orientation)
        get higher scores. Uses average distance for smooth diversity.
        """
        n_cameras = len(self.all_cameras)
        
        if not self.recent_selections:
            # No history yet, use k-NN diversity as initial scores
            k = min(self.k_neighbors, n_cameras - 1)
            base_scores = np.zeros(n_cameras)
            for i in range(n_cameras):
                dists = np.linalg.norm(
                    self.pose_features_scaled - self.pose_features_scaled[i],
                    axis=1
                )
                nearest_dists = np.sort(dists)[1:k+1]
                base_scores[i] = np.mean(nearest_dists) if len(nearest_dists) > 0 else 0.0
        else:
            # Get pose features of recently selected cameras
            recent_indices = self.recent_selections[-self.recency_window:]
            recent_features = self.pose_features_scaled[recent_indices]
            
            # Average distance to all recent selections in pose space
            # This gives smooth diversity - cameras far from ALL recent selections score high
            base_scores = np.mean(
                np.linalg.norm(
                    self.pose_features_scaled[:, None, :] - recent_features[None, :, :],
                    axis=2
                ),
                axis=1
            )
        
        # Normalize to [0, 1]
        if base_scores.max() > base_scores.min():
            base_scores = (base_scores - base_scores.min()) / (base_scores.max() - base_scores.min())
        else:
            base_scores = np.ones(n_cameras)
        
        # Apply cumulative penalty (optional)
        if self.use_cumulative_penalty and self.selection_counts is not None:
            alpha = 0.1  # Penalty strength
            penalty = 1.0 / (1.0 + alpha * self.selection_counts)
            scores = base_scores * penalty
        else:
            scores = base_scores
        
        return scores

    def compute_probabilities(self, gaussians, iteration: int) -> Dict[int, float]:
        """
        Compute selection probabilities.

        Args:
            gaussians: Current Gaussian model (unused)
            iteration: Current training iteration

        Returns:
            Dictionary mapping camera uid to probability
        """
        if self.mode == 'static':
            return self.selection_probabilities
        
        # Dynamic mode: compute distance-based scores
        scores = self._compute_distance_to_selected_scores()
        
        # Convert to probabilities with temperature
        probabilities = softmax(scores / self.temperature)
        
        return {
            cam.uid: float(probabilities[i])
            for i, cam in enumerate(self.all_cameras)
        }
    
    def log_selection(self, cam, score: float, iteration: int) -> None:
        """Record selection for dynamic diversity tracking."""
        super().log_selection(cam, score, iteration)
        
        # Track selection for dynamic mode
        idx = self.cam_uid_to_idx.get(cam.uid)
        if idx is not None:
            self.recent_selections.append(idx)
            if self.selection_counts is not None:
                self.selection_counts[idx] += 1
        
        # Trim recency buffer
        if len(self.recent_selections) > self.recency_window * 2:
            self.recent_selections = self.recent_selections[-self.recency_window:]
