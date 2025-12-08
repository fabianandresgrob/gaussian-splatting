"""
Fixed probability view selection based on pose heuristics.

This selector computes probabilities once at initialization based on camera pose
properties and keeps them fixed throughout training.
"""

import numpy as np
import torch
from typing import Dict, List
from scipy.special import softmax
from .selector import ViewSelector


class FixedProbabilitySelector(ViewSelector):
    """
    Selector that computes fixed probabilities based on camera pose heuristics.

    Heuristics include:
    - Distance from scene center (prioritize diverse viewpoints)
    - Spatial diversity (avoid over-sampling clustered cameras)

    Probabilities are computed once and remain fixed during training.
    """

    def __init__(self, config: dict = None, log_dir: str = None, verbose: bool = False, seed: int = None):
        """
        Initialize the fixed probability selector.

        Args:
            config: Configuration dictionary with optional keys:
                - temperature (float): Softmax temperature for probability distribution.
                  Higher values make distribution more uniform. Default: 1.0
                - distance_weight (float): Weight for distance heuristic. Default: 0.5
                - diversity_weight (float): Weight for diversity heuristic. Default: 0.5
            log_dir: Directory to save selection logs
            verbose: If True, print detailed information
            seed: Random seed for reproducibility
        """
        super().__init__(config or {}, log_dir, verbose, seed=seed)
        self.temperature = self.config.get('temperature', 1.0)
        self.distance_weight = self.config.get('distance_weight', 0.5)
        self.diversity_weight = self.config.get('diversity_weight', 0.5)
        self.fixed_probabilities = None

    def initialize(self, all_cameras: List) -> None:
        """
        Compute fixed probabilities based on camera poses.

        Args:
            all_cameras: List of all available Camera objects
        """
        super().initialize(all_cameras)
        if self.verbose:
            print(f"[FixedProbabilitySelector] Initializing with {len(all_cameras)} cameras")
            print(f"[FixedProbabilitySelector] Temperature: {self.temperature}")
            print(f"[FixedProbabilitySelector] Distance weight: {self.distance_weight}")
            print(f"[FixedProbabilitySelector] Diversity weight: {self.diversity_weight}")

        # Extract camera positions
        positions = []
        for cam in all_cameras:
            # Camera center is stored as a torch tensor
            pos = cam.camera_center.cpu().numpy()
            positions.append(pos)
        positions = np.array(positions)

        # Compute scene center
        scene_center = np.mean(positions, axis=0)

        # Heuristic 1: Distance from scene center
        # Cameras far from center provide different perspectives
        distances = np.linalg.norm(positions - scene_center, axis=1)
        # Normalize to [0, 1]
        if distances.max() > 0:
            distance_scores = distances / distances.max()
        else:
            distance_scores = np.ones_like(distances)

        # Heuristic 2: Spatial diversity
        # For each camera, compute average distance to k nearest neighbors
        # Cameras in sparse regions get higher scores
        k = min(5, len(all_cameras) - 1)
        diversity_scores = np.zeros(len(all_cameras))

        for i, pos_i in enumerate(positions):
            # Compute distances to all other cameras
            dists = np.linalg.norm(positions - pos_i, axis=1)
            # Sort and take k nearest (excluding self at index 0)
            nearest_dists = np.sort(dists)[1:k+1]
            # Average distance to nearest neighbors
            diversity_scores[i] = np.mean(nearest_dists)

        # Normalize diversity scores
        if diversity_scores.max() > 0:
            diversity_scores = diversity_scores / diversity_scores.max()
        else:
            diversity_scores = np.ones_like(diversity_scores)

        # Combine heuristics
        combined_scores = (self.distance_weight * distance_scores +
                          self.diversity_weight * diversity_scores)

        # Convert scores to probabilities using softmax with temperature
        probabilities = softmax(combined_scores / self.temperature)

        # Store as dictionary
        self.fixed_probabilities = {
            cam.uid: float(probabilities[i])
            for i, cam in enumerate(all_cameras)
        }

        self.initialized = True

        if self.verbose:
            print(f"[FixedProbabilitySelector] Probability range: "
                  f"{min(probabilities):.4f} - {max(probabilities):.4f}")
            print(f"[FixedProbabilitySelector] Probability std: {np.std(probabilities):.4f}")

    def compute_probabilities(self, gaussians, iteration: int) -> Dict[int, float]:
        """
        Return the pre-computed fixed probabilities.

        Args:
            gaussians: Current Gaussian model (unused)
            iteration: Current training iteration (unused)

        Returns:
            Dictionary mapping camera uid to probability
        """
        return self.fixed_probabilities
