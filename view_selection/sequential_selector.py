"""
Sequential view selection strategy (baseline).

This selector iterates through cameras in a fixed order based on their index.
When it reaches the end of the list, it wraps around to the beginning.
This is a deterministic baseline with no randomness.
"""

from typing import Dict, List
from .selector import ViewSelector


class SequentialSelector(ViewSelector):
    """
    Sequential baseline selector that iterates through cameras in fixed order.

    Cameras are sorted by their uid and selected one by one in that order.
    When the end of the list is reached, selection wraps back to the start.

    This is a purely deterministic strategy - no randomness involved.
    Useful as a baseline to compare against random and adaptive strategies.
    """

    def __init__(self, config: dict = None, log_dir: str = None, verbose: bool = False, seed: int = None):
        """
        Initialize the sequential selector.

        Args:
            config: Configuration dictionary (unused for sequential selection)
            log_dir: Directory to save selection logs
            verbose: If True, print selection information
            seed: Random seed (unused - this selector is deterministic)
        """
        super().__init__(config or {}, log_dir, verbose, seed=seed)
        self.current_index = 0
        self.camera_order = []  # Sorted list of camera uids

    def initialize(self, all_cameras: List) -> None:
        """
        Initialize the selector with camera information.

        Args:
            all_cameras: List of all available Camera objects
        """
        super().initialize(all_cameras)

        # Sort cameras by uid for deterministic ordering
        self.camera_order = sorted(all_cameras, key=lambda cam: cam.uid)
        self.current_index = 0
        self.initialized = True

        self.logger.info(f"Initialized with {len(self.camera_order)} cameras in sequential order")
        if self.verbose:
            self.logger.debug(f"Camera order (by uid): {[cam.uid for cam in self.camera_order]}")

    def compute_probabilities(self, gaussians, iteration: int) -> Dict[int, float]:
        """
        Compute probabilities for sequential selection.

        Only the current camera in the sequence gets probability 1.0,
        all others get 0.0. This makes selection deterministic.

        Args:
            gaussians: Current Gaussian model (unused)
            iteration: Current training iteration (unused)

        Returns:
            Dictionary mapping camera uid to probability (1.0 for current, 0.0 for others)
        """
        current_cam = self.camera_order[self.current_index]
        probabilities = {cam.uid: 0.0 for cam in self.all_cameras}
        probabilities[current_cam.uid] = 1.0
        return probabilities

    def select_view(self, gaussians, iteration: int):
        """
        Select the next camera in sequence.

        Overrides base class to implement deterministic sequential selection
        instead of probability-based sampling.

        Args:
            gaussians: Current Gaussian model (unused)
            iteration: Current training iteration

        Returns:
            Selected Camera object
        """
        if not self.initialized:
            raise RuntimeError("ViewSelector must be initialized before use. Call initialize() first.")

        # Get current camera
        selected_cam = self.camera_order[self.current_index]

        # Log the selection
        self.log_selection(selected_cam, 1.0, iteration)

        # Advance to next camera (wrap around at end)
        self.current_index = (self.current_index + 1) % len(self.camera_order)

        return selected_cam
