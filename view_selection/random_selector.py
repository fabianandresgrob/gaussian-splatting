"""
Random view selection strategy (baseline).

This replicates the original behavior where cameras are selected uniformly at random.
"""

import numpy as np
from typing import Dict, List
from .selector import ViewSelector


class RandomSelector(ViewSelector):
    """
    Baseline selector that assigns uniform probability to all cameras.
    """

    def __init__(self, config: dict = None, log_dir: str = None, verbose: bool = False, seed: int = None):
        """
        Initialize the random selector.

        Args:
            config: Configuration dictionary (unused for random selection)
            log_dir: Directory to save selection logs
            verbose: If True, print selection information
            seed: Random seed for reproducibility
        """
        super().__init__(config or {}, log_dir, verbose, seed=seed)
        self.num_cameras = 0

    def initialize(self, all_cameras: List) -> None:
        """
        Initialize the selector with camera information.

        Args:
            all_cameras: List of all available Camera objects
        """
        super().initialize(all_cameras)
        self.num_cameras = len(all_cameras)
        self.initialized = True

        if self.verbose:
            print(f"[RandomSelector] Initialized with {self.num_cameras} cameras")
            print(f"[RandomSelector] Each camera has probability {1.0/self.num_cameras:.4f}")

    def compute_probabilities(self, gaussians, iteration: int) -> Dict[int, float]:
        """
        Compute uniform probabilities for all cameras.

        Args:
            gaussians: Current Gaussian model (unused)
            iteration: Current training iteration (unused)

        Returns:
            Dictionary mapping camera uid to uniform probability
        """
        # All cameras get equal probability
        uniform_prob = 1.0 / len(self.all_cameras)
        probabilities = {cam.uid: uniform_prob for cam in self.all_cameras}

        return probabilities
