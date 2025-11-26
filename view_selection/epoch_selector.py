"""
Epoch-based view selection with dynamic probability adjustment.

This selector extends FixedProbabilitySelector by periodically recomputing probabilities
to penalize recently selected cameras, ensuring all views get seen while still
prioritizing informative ones.
"""

import numpy as np
from typing import Dict, List
from scipy.special import softmax
from .heuristic_selector import FixedProbabilitySelector


class EpochBasedSelector(FixedProbabilitySelector):
    """
    Selector that recomputes probabilities every "epoch" (N selections, where N = number of cameras).

    At each recomputation:
    - Start with base heuristic scores (from FixedProbabilitySelector)
    - Apply penalty based on recent selection counts
    - Adjusted score = base_score / (1 + selection_count_this_epoch)

    This ensures all cameras get sampled while maintaining preference for informative views.
    """

    def __init__(self, config: dict = None, log_dir: str = None, verbose: bool = False, seed: int = None):
        """
        Initialize the epoch-based selector.

        Args:
            config: Configuration dictionary with optional keys:
                - temperature (float): Softmax temperature. Default: 1.0
                - distance_weight (float): Weight for distance heuristic. Default: 0.5
                - diversity_weight (float): Weight for diversity heuristic. Default: 0.5
                - penalty_strength (float): How much to penalize recently selected cameras.
                  Higher values encourage more uniform sampling. Default: 1.0
            log_dir: Directory to save selection logs
            verbose: If True, print detailed information
            seed: Random seed for reproducibility
        """
        super().__init__(config, log_dir, verbose, seed=seed)
        self.penalty_strength = self.config.get('penalty_strength', 1.0)
        self.base_scores = None  # Heuristic scores before penalty
        self.epoch_selection_counts = {}  # Selection counts within current epoch
        self.selections_this_epoch = 0
        self.num_cameras = 0
        self.epoch_number = 0

    def initialize(self, all_cameras: List) -> None:
        """
        Compute base heuristic scores.

        Args:
            all_cameras: List of all available Camera objects
        """
        # Call parent to compute fixed probabilities
        super().initialize(all_cameras)

        self.num_cameras = len(all_cameras)

        # Extract camera positions for computing base scores
        positions = []
        for cam in all_cameras:
            pos = cam.camera_center.cpu().numpy()
            positions.append(pos)
        positions = np.array(positions)

        # Compute scene center
        scene_center = np.mean(positions, axis=0)

        # Compute base heuristic scores (same as FixedProbabilitySelector)
        distances = np.linalg.norm(positions - scene_center, axis=1)
        if distances.max() > 0:
            distance_scores = distances / distances.max()
        else:
            distance_scores = np.ones_like(distances)

        k = min(5, len(all_cameras) - 1)
        diversity_scores = np.zeros(len(all_cameras))
        for i, pos_i in enumerate(positions):
            dists = np.linalg.norm(positions - pos_i, axis=1)
            nearest_dists = np.sort(dists)[1:k+1]
            diversity_scores[i] = np.mean(nearest_dists)

        if diversity_scores.max() > 0:
            diversity_scores = diversity_scores / diversity_scores.max()
        else:
            diversity_scores = np.ones_like(diversity_scores)

        combined_scores = (self.distance_weight * distance_scores +
                          self.diversity_weight * diversity_scores)

        # Store base scores
        self.base_scores = {
            cam.uid: float(combined_scores[i])
            for i, cam in enumerate(all_cameras)
        }

        # Initialize epoch counters
        self.epoch_selection_counts = {cam.uid: 0 for cam in all_cameras}
        self.selections_this_epoch = 0
        self.epoch_number = 0

        if self.verbose:
            print(f"[EpochBasedSelector] Initialized with {self.num_cameras} cameras")
            print(f"[EpochBasedSelector] Epoch size: {self.num_cameras} selections")
            print(f"[EpochBasedSelector] Penalty strength: {self.penalty_strength}")

    def _start_new_epoch(self):
        """Start a new epoch by resetting counters."""
        self.epoch_selection_counts = {cam.uid: 0 for cam in self.all_cameras}
        self.selections_this_epoch = 0
        self.epoch_number += 1

        if self.verbose:
            print(f"\n[EpochBasedSelector] Starting epoch {self.epoch_number}")

    def compute_probabilities(self, gaussians, iteration: int) -> Dict[int, float]:
        """
        Compute probabilities with recency penalty.

        Args:
            gaussians: Current Gaussian model (unused)
            iteration: Current training iteration

        Returns:
            Dictionary mapping camera uid to probability
        """
        # Check if we need to start a new epoch
        # An epoch ends when we've made N selections (where N = num_cameras)
        if self.selections_this_epoch >= self.num_cameras:
            self._start_new_epoch()

        # Calculate current scores: base_score / (1 + penalty * selection_count)
        current_scores = []
        uids = []

        for cam in self.all_cameras:
            uid = cam.uid
            base_score = self.base_scores[uid]
            count = self.epoch_selection_counts[uid]
            
            # Apply penalty
            # If count is 0, score is base_score
            # If count is high, score decreases
            adjusted_score = base_score / (1.0 + self.penalty_strength * count)
            
            current_scores.append(adjusted_score)
            uids.append(uid)

        current_scores = np.array(current_scores)
        
        # Apply softmax to get probabilities
        probs = softmax(current_scores / self.temperature)
        
        return {uid: float(prob) for uid, prob in zip(uids, probs)}

    def log_selection(self, cam, score: float, iteration: int) -> None:
        """
        Record selection and update epoch counters.

        Args:
            cam: Selected Camera object
            score: Probability that led to this selection
            iteration: Current training iteration
        """
        # Call parent logging
        super().log_selection(cam, score, iteration)

        # Update epoch counters
        self.epoch_selection_counts[cam.uid] += 1
        self.selections_this_epoch += 1

        if self.verbose and self.selections_this_epoch % 100 == 0:
            print(f"[EpochBasedSelector] Epoch {self.epoch_number}: "
                  f"{self.selections_this_epoch}/{self.num_cameras} selections")
