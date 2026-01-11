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
    - Distance from camera centroid (cameras far from center = high score)
    - Average distance to other cameras (spatial diversity)
    - Distance to recently selected cameras (dynamic diversity)

    Modes:
    - 'static': Probabilities computed once at init (original behavior)
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
                - distance_weight (float): Weight for distance heuristic (static mode). Default: 0.5
                - diversity_weight (float): Weight for diversity heuristic (static mode). Default: 0.5
                - recency_window (int): Number of recent selections to consider (dynamic mode). Default: 500
                - use_cumulative_penalty (bool): Apply penalty based on total selection count. Default: True
            log_dir: Directory to save selection logs
            verbose: If True, print detailed information
            seed: Random seed for reproducibility
        """
        super().__init__(config or {}, log_dir, verbose, seed=seed)
        self.mode = self.config.get('mode', 'distance_to_selected')
        self.temperature = self.config.get('temperature', 1.0)
        self.distance_weight = self.config.get('distance_weight', 0.5)
        self.diversity_weight = self.config.get('diversity_weight', 0.5)
        self.recency_window = self.config.get('recency_window', 500)
        # NOTE: Disabled by default - this tracks by camera uid, not by pose similarity,
        # so duplicates with different uids would be tracked separately. The recency-based
        # approach using pose distance is the fair mechanism for diversity.
        self.use_cumulative_penalty = self.config.get('use_cumulative_penalty', False)
        
        # Static mode storage
        self.selection_probabilities = None
        
        # Dynamic mode storage
        self.positions = None  # (N, 3) array of camera positions
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
        self.logger.debug(f"Temperature: {self.temperature}")
        
        # Extract camera positions
        positions = []
        for cam in all_cameras:
            pos = cam.camera_center.cpu().numpy()
            positions.append(pos)
        self.positions = np.array(positions)
        
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
        """Compute fixed probabilities based on pose heuristics (original behavior)."""
        self.logger.debug(f"Distance weight: {self.distance_weight}")
        self.logger.debug(f"Diversity weight: {self.diversity_weight}")
        
        # Compute scene center
        scene_center = np.mean(self.positions, axis=0)

        # Heuristic 1: Distance from scene center
        distances = np.linalg.norm(self.positions - scene_center, axis=1)
        if distances.max() > 0:
            distance_scores = distances / distances.max()
        else:
            distance_scores = np.ones(len(self.positions))

        # Heuristic 2: Spatial diversity (average distance to k nearest neighbors)
        k = min(5, len(self.all_cameras) - 1)
        diversity_scores = np.zeros(len(self.all_cameras))

        for i, pos_i in enumerate(self.positions):
            dists = np.linalg.norm(self.positions - pos_i, axis=1)
            nearest_dists = np.sort(dists)[1:k+1]
            diversity_scores[i] = np.mean(nearest_dists)

        if diversity_scores.max() > 0:
            diversity_scores = diversity_scores / diversity_scores.max()
        else:
            diversity_scores = np.ones_like(diversity_scores)

        # Combine heuristics
        combined_scores = (self.distance_weight * distance_scores +
                          self.diversity_weight * diversity_scores)

        # Convert to probabilities
        probabilities = softmax(combined_scores / self.temperature)

        self.selection_probabilities = {
            cam.uid: float(probabilities[i])
            for i, cam in enumerate(self.all_cameras)
        }

        self.logger.info(f"Probability range: {min(probabilities):.4f} - {max(probabilities):.4f}")

    def _compute_distance_to_selected_scores(self) -> np.ndarray:
        """
        Compute scores based on distance to recently selected cameras.
        
        Cameras far from recent selections get higher scores.
        Applies cumulative penalty to down-weight frequently selected cameras.
        """
        n_cameras = len(self.all_cameras)
        
        if not self.recent_selections:
            # No history yet, use distance from centroid
            centroid = np.mean(self.positions, axis=0, keepdims=True)
            base_scores = np.linalg.norm(self.positions - centroid, axis=1)
        else:
            # Get positions of recently selected cameras
            recent_indices = self.recent_selections[-self.recency_window:]
            recent_positions = self.positions[recent_indices]
            
            # For each camera, compute minimum distance to any recent selection
            # (prioritize cameras far from ALL recent selections)
            scores = np.zeros(n_cameras)
            for i in range(n_cameras):
                distances = np.linalg.norm(recent_positions - self.positions[i], axis=1)
                # Use minimum distance (camera is "close" if close to ANY recent selection)
                scores[i] = distances.min()
            
            base_scores = scores
        
        # Normalize to [0, 1]
        if base_scores.max() > base_scores.min():
            base_scores = (base_scores - base_scores.min()) / (base_scores.max() - base_scores.min())
        else:
            base_scores = np.ones(n_cameras)
        
        # Apply cumulative penalty
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
