"""
Deterministic loss-based view selection strategy.

This selector computes the loss for ALL cameras at each iteration and selects
the camera with the highest loss. This is fully deterministic with no sampling.

Note: This is computationally expensive as it requires rendering all views
at each iteration to compute their losses.
"""

import torch
from typing import Dict, List, Optional
from .selector import ViewSelector


class DeterministicMaxLossSelector(ViewSelector):
    """
    Deterministic selector that always picks the camera with highest loss.

    At each iteration:
    1. Renders all cameras
    2. Computes loss for each
    3. Selects the camera with maximum loss

    This is a greedy strategy focusing training on the hardest views.
    Computationally expensive but fully deterministic.

    Config parameters:
        - None (this strategy has no hyperparameters)

    Note: Requires the render function and loss function to be passed
    during initialization or set before use.
    """

    def __init__(self, config: dict = None, log_dir: str = None, verbose: bool = False, seed: int = None):
        """
        Initialize the deterministic max-loss selector.

        Args:
            config: Configuration dictionary (unused)
            log_dir: Directory to save selection logs
            verbose: If True, print detailed information
            seed: Random seed (unused - this selector is deterministic)
        """
        super().__init__(config or {}, log_dir, verbose, seed)

        # These must be set before use via set_render_context()
        self._render_fn = None
        self._pipe = None
        self._background = None
        self._loss_fn = None

        # Cache losses for logging/analysis
        self.last_losses = {}  # uid -> loss from last compute_all_losses call

    def set_render_context(self, render_fn, pipe, background, loss_fn) -> None:
        """
        Set the rendering context needed to compute losses.

        This must be called before using the selector.

        Args:
            render_fn: The render function (from gaussian_renderer)
            pipe: Pipeline parameters
            background: Background color tensor
            loss_fn: Loss function that takes (rendered_image, gt_image) and returns scalar loss
        """
        self._render_fn = render_fn
        self._pipe = pipe
        self._background = background
        self._loss_fn = loss_fn

    def initialize(self, all_cameras: List) -> None:
        """
        Initialize the selector with camera information.

        Args:
            all_cameras: List of all available Camera objects
        """
        super().initialize(all_cameras)
        self.initialized = True
        self.logger.info(f"Initialized with {len(all_cameras)} cameras for deterministic max-loss selection")

    def compute_all_losses(self, gaussians) -> Dict[int, float]:
        """
        Compute the loss for all cameras by rendering each one.

        Args:
            gaussians: Current Gaussian model

        Returns:
            Dictionary mapping camera uid to its loss value
        """
        if self._render_fn is None:
            raise RuntimeError(
                "Render context not set. Call set_render_context() before using this selector."
            )

        losses = {}

        with torch.no_grad():
            for cam in self.all_cameras:
                # Render the view
                render_pkg = self._render_fn(cam, gaussians, self._pipe, self._background)
                image = render_pkg["render"]

                # Get ground truth
                gt_image = cam.original_image.cuda()

                # Compute loss
                loss = self._loss_fn(image, gt_image)

                # Handle tensor loss
                if hasattr(loss, 'item'):
                    loss = loss.item()

                losses[cam.uid] = loss

        self.last_losses = losses
        return losses

    def compute_probabilities(self, gaussians, iteration: int) -> Dict[int, float]:
        """
        Compute probabilities based on losses (max loss gets probability 1.0).

        This computes losses for all cameras and assigns probability 1.0 to
        the camera with the highest loss, and 0.0 to all others.

        Args:
            gaussians: Current Gaussian model
            iteration: Current training iteration

        Returns:
            Dictionary mapping camera uid to probability
        """
        # Compute losses for all cameras
        losses = self.compute_all_losses(gaussians)

        # Find camera with maximum loss
        max_uid = max(losses, key=losses.get)
        max_loss = losses[max_uid]

        # Assign probabilities: 1.0 to max, 0.0 to others
        probabilities = {uid: 0.0 for uid in losses}
        probabilities[max_uid] = 1.0

        if self.verbose and iteration % 100 == 0:
            min_loss = min(losses.values())
            self.logger.debug(
                f"Iter {iteration}: Computed {len(losses)} losses, "
                f"range=[{min_loss:.6f}, {max_loss:.6f}], "
                f"selected uid={max_uid}"
            )

        return probabilities

    def select_view(self, gaussians, iteration: int):
        """
        Select the camera with the highest loss.

        Overrides base class to implement deterministic selection
        based on computed losses.

        Args:
            gaussians: Current Gaussian model
            iteration: Current training iteration

        Returns:
            Selected Camera object (the one with highest loss)
        """
        if not self.initialized:
            raise RuntimeError("ViewSelector must be initialized before use. Call initialize() first.")

        # Compute losses and get probabilities (which will be 1.0 for max loss camera)
        probabilities = self.compute_probabilities(gaussians, iteration)

        # Find the camera with probability 1.0 (max loss)
        selected_uid = max(probabilities, key=probabilities.get)

        # Find the camera object
        selected_cam = None
        for cam in self.all_cameras:
            if cam.uid == selected_uid:
                selected_cam = cam
                break

        if selected_cam is None:
            raise RuntimeError(f"Camera with uid {selected_uid} not found")

        # Log the selection with the actual loss value
        loss_value = self.last_losses.get(selected_uid, 0.0)
        self.log_selection(selected_cam, loss_value, iteration)

        return selected_cam

    def get_loss_statistics(self) -> Dict:
        """
        Get statistics about the last computed losses.

        Returns:
            Dictionary with loss statistics
        """
        if not self.last_losses:
            return {}

        losses = list(self.last_losses.values())
        import numpy as np
        return {
            'mean_loss': float(np.mean(losses)),
            'std_loss': float(np.std(losses)),
            'min_loss': float(np.min(losses)),
            'max_loss': float(np.max(losses)),
            'num_cameras': len(losses),
        }
