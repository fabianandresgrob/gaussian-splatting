"""
Stack-based view selection (default baseline).

This selector maintains an internal shuffled queue of camera UIDs and returns
one camera per selection by popping from the queue. When the queue is empty it
is refilled with a new random permutation of the available cameras.

This replicates the original 3DGS baseline behavior where views were drawn from a
`viewpoint_stack` and reinitialized when empty. Every view is seen exactly once
per "epoch", guaranteeing uniform coverage.
"""

import numpy as np
from typing import Dict, List, Optional
from .selector import ViewSelector


class StackBasedSelector(ViewSelector):
    """Stack-based sampling without replacement (default 3DGS baseline).

    Process:
    1. Shuffle N cameras into stack
    2. Pop one per iteration
    3. Refill when empty

    Every view is seen exactly once per "epoch" - guaranteed uniform coverage.

    Config options:
      - seed (int, optional): RNG seed for deterministic shuffles.
    """

    def __init__(self, config: dict = None, log_dir: Optional[str] = None, verbose: bool = False, seed: int = None):
        super().__init__(config or {}, log_dir, verbose, seed=seed)
        # Note: self.rng is already set up by the parent class with the seed

        # Internal queue of uids (acts like the old viewpoint_stack)
        self._queue = []  # type: List[int]
        self._uid_to_cam = {}

    def initialize(self, all_cameras: List) -> None:
        """Prepare internal mappings and prime the queue.

        Args:
            all_cameras: list of Camera objects
        """
        super().initialize(all_cameras)
        # Build mapping from uid -> camera for fast lookup
        self._uid_to_cam = {cam.uid: cam for cam in all_cameras}

        # Initialize queue
        self._refill_queue(list(self._uid_to_cam.keys()))

        self.initialized = True

        self.logger.info(f"Initialized with {len(all_cameras)} cameras, seed={self.seed}")

    def _refill_queue(self, uids: List[int]) -> None:
        """Shuffle and refill the internal queue."""
        # Use in-place shuffle on a copy to avoid mutating input
        uids_copy = list(uids)
        # If rng is numpy module (no seed), it provides shuffle; if RandomState, also shuffle works
        self.rng.shuffle(uids_copy)
        # Use list as a stack: pop() from end
        self._queue = uids_copy

    def compute_probabilities(self, gaussians, iteration: int) -> Dict[int, float]:
        """Return uniform probabilities (kept for API compatibility).

        The selection is actually handled by overriding select_view(), but returning
        uniform probabilities keeps compatibility with any code that inspects them.
        """
        uniform = 1.0 / len(self.all_cameras) if len(self.all_cameras) > 0 else 0.0
        return {cam.uid: float(uniform) for cam in self.all_cameras}

    def select_view(self, gaussians, iteration: int):
        """Select next camera by popping from the internal shuffled queue.

        If the queue is empty it is refilled with a new shuffle of the current
        set of camera UIDs (useful if cameras list changes during training).
        """
        if not self.initialized:
            raise RuntimeError("ViewSelector must be initialized before use. Call initialize() first.")

        if not self._queue:
            self._refill_queue(list(self._uid_to_cam.keys()))

        selected_uid = self._queue.pop()

        # Lookup camera object
        selected_cam = self._uid_to_cam[selected_uid]

        # Log selection with a uniform probability for bookkeeping
        prob = 1.0 / len(self.all_cameras) if len(self.all_cameras) > 0 else 0.0
        self.log_selection(selected_cam, prob, iteration)

        return selected_cam
