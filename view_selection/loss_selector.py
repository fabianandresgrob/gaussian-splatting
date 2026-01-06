"""
Loss-based view selection strategy.

This selector tracks rendering loss for each camera and biases sampling toward
cameras with higher average loss, on the theory that these views are harder
to fit and need more training attention.
"""

import numpy as np
from typing import Dict, List
from scipy.special import softmax
from .selector import ViewSelector


class LossBasedSelector(ViewSelector):
    """
    Selector that prioritizes cameras with higher rendering loss.

    Uses Exponential Moving Average (EMA) to track loss per camera:
    - Cameras with higher average loss get higher sampling probability
    - EMA smooths loss estimates over time with configurable decay
    - Falls back to uniform sampling for cameras without loss history
    - Once min_samples_before_bias is reached, switches to loss-biased sampling

    Config parameters:
        - ema_decay (float): EMA decay factor for loss tracking. Default: 0.99
          Higher values = more weight on historical losses
        - temperature (float): Softmax temperature for probability calculation. Default: 1.0
          Higher values = more uniform, lower values = more peaked
        - min_samples_before_bias (int): Minimum samples per camera before applying
          loss-based bias. Default: 5
          Until this threshold is met, uses uniform sampling
    """

    def __init__(self, config: dict = None, log_dir: str = None, verbose: bool = False, seed: int = None):
        """
        Initialize the loss-based selector.

        Args:
            config: Configuration dictionary with optional keys:
                - ema_decay (float): EMA decay for loss tracking. Default: 0.99
                - temperature (float): Softmax temperature. Default: 1.0
                - min_samples_before_bias (int): Min samples before biasing. Default: 5
            log_dir: Directory to save selection logs
            verbose: If True, print detailed information
            seed: Random seed for reproducibility
        """
        super().__init__(config, log_dir, verbose, seed)

        # Configuration parameters
        self.ema_decay = self.config.get('ema_decay', 0.99)
        self.temperature = self.config.get('temperature', 1.0)
        self.min_samples_before_bias = self.config.get('min_samples_before_bias', 5)

        # Loss tracking
        self.ema_losses = {}  # Dict mapping camera uid to EMA loss
        self.loss_sample_counts = {}  # Dict mapping camera uid to number of loss updates

        if self.verbose:
            self.logger.debug(f"Configuration: ema_decay={self.ema_decay}, "
                            f"temperature={self.temperature}, min_samples={self.min_samples_before_bias}")

    def initialize(self, all_cameras: List) -> None:
        """
        Initialize loss tracking structures.

        Args:
            all_cameras: List of all available Camera objects
        """
        super().initialize(all_cameras)

        # Initialize loss tracking for each camera
        for cam in all_cameras:
            self.ema_losses[cam.uid] = 0.0
            self.loss_sample_counts[cam.uid] = 0

        self.initialized = True

        self.logger.info(f"Initialized with {len(all_cameras)} cameras")

    def update_loss(self, camera, loss: float) -> None:
        """
        Update the EMA loss for a specific camera.

        This method should be called from the training loop after rendering
        a view and computing the loss.

        Args:
            camera: Camera object that was just rendered
            loss: Loss value (float or tensor) from rendering this camera

        Example:
            >>> # In training loop after loss computation:
            >>> loss = l1_loss + ssim_loss
            >>> selector.update_loss(viewpoint_cam, loss.item())
        """
        # Convert loss to float if it's a tensor
        if hasattr(loss, 'item'):
            loss = loss.item()

        uid = camera.uid

        # Initialize EMA with the first actual observation for this camera
        # Note: initialize() pre-populates ema_losses with 0.0 for all cameras
        # Using EMA from 0.0 would shrink the first real loss by (1-ema_decay)
        # Checking the sample count avoids that and treats the first sample as the baseline
        prev_count = self.loss_sample_counts.get(uid, 0)
        if uid not in self.ema_losses or prev_count == 0:
            self.ema_losses[uid] = float(loss)
            self.loss_sample_counts[uid] = 1
        else:
            # Update EMA: new_ema = decay * old_ema + (1 - decay) * new_value
            old_ema = self.ema_losses[uid]
            self.ema_losses[uid] = self.ema_decay * old_ema + (1 - self.ema_decay) * float(loss)
            self.loss_sample_counts[uid] = prev_count + 1

        if self.verbose and self.loss_sample_counts[uid] % 100 == 0:
            self.logger.debug(f"Camera {uid} ({camera.image_name}): "
                  f"EMA loss = {self.ema_losses[uid]:.6f} "
                  f"(samples: {self.loss_sample_counts[uid]})")

    def compute_probabilities(self, gaussians, iteration: int) -> Dict[int, float]:
        """
        Compute sampling probabilities based on EMA losses.

        Strategy:
        - If not enough samples collected yet, use uniform sampling
        - Otherwise, cameras with higher EMA loss get higher probability
        - Use softmax to convert loss scores to valid probabilities

        Args:
            gaussians: Current Gaussian model (unused)
            iteration: Current training iteration

        Returns:
            Dictionary mapping camera uid to sampling probability
        """
        num_cameras = len(self.all_cameras)

        # Check if we have enough samples for all cameras
        min_samples = min(self.loss_sample_counts.values()) if self.loss_sample_counts else 0
        use_uniform = min_samples < self.min_samples_before_bias

        if use_uniform:
            # Uniform sampling as fallback
            uniform_prob = 1.0 / num_cameras
            probabilities = {cam.uid: uniform_prob for cam in self.all_cameras}

            if self.verbose and iteration % 1000 == 0:
                self.logger.debug(f"Iter {iteration}: Uniform sampling "
                      f"(min samples: {min_samples}/{self.min_samples_before_bias})")
        else:
            # Loss-based sampling
            losses = []
            uids = []

            for cam in self.all_cameras:
                uid = cam.uid
                # Higher loss = higher score for sampling
                losses.append(self.ema_losses[uid])
                uids.append(uid)

            losses = np.array(losses)

            # Normalize losses to [0, 1] range for more stable softmax
            # This prevents scale issues when comparing across different scenes
            if losses.max() > losses.min():
                losses_normalized = (losses - losses.min()) / (losses.max() - losses.min())
            else:
                losses_normalized = losses

            # Apply softmax with temperature to get probabilities
            # Higher temperature -> more uniform, lower -> more peaked
            probs = softmax(losses_normalized / self.temperature)

            probabilities = {uid: float(prob) for uid, prob in zip(uids, probs)}

            if self.verbose and iteration % 1000 == 0:
                max_loss_idx = np.argmax(losses)
                min_loss_idx = np.argmin(losses)
                self.logger.debug(f"Iter {iteration}: Loss-based sampling, "
                      f"loss=[{losses[min_loss_idx]:.6f}, {losses[max_loss_idx]:.6f}], "
                      f"prob=[{probs.min():.4f}, {probs.max():.4f}]")

        return probabilities

    def get_loss_statistics(self) -> Dict:
        """
        Get statistics about loss tracking.

        Returns:
            Dictionary with loss statistics including mean, std, min, max losses
        """
        if not self.ema_losses:
            return {}

        losses = list(self.ema_losses.values())
        sample_counts = list(self.loss_sample_counts.values())

        stats = {
            'mean_ema_loss': float(np.mean(losses)),
            'std_ema_loss': float(np.std(losses)),
            'min_ema_loss': float(np.min(losses)),
            'max_ema_loss': float(np.max(losses)),
            'mean_sample_count': float(np.mean(sample_counts)),
            'min_sample_count': int(np.min(sample_counts)),
            'max_sample_count': int(np.max(sample_counts)),
            'total_loss_updates': int(np.sum(sample_counts)),
        }

        return stats

    def log_statistics(self, iteration: int) -> None:
        """
        Print loss and selection statistics.

        Args:
            iteration: Current training iteration
        """
        # Get base selection statistics
        selection_stats = self.get_selection_statistics()

        # Get loss statistics
        loss_stats = self.get_loss_statistics()

        self.logger.info(f"Statistics at iteration {iteration}:")
        self.logger.info(f"  Selections: {selection_stats.get('total_selections', 0)}, "
                        f"unique cameras: {selection_stats.get('unique_cameras', 0)}")

        if loss_stats:
            self.logger.info(f"  Loss: mean={loss_stats['mean_ema_loss']:.6f} ± {loss_stats['std_ema_loss']:.6f}, "
                  f"range=[{loss_stats['min_ema_loss']:.6f}, {loss_stats['max_ema_loss']:.6f}]")
            self.logger.info(f"  Samples: count=[{loss_stats['min_sample_count']}, {loss_stats['max_sample_count']}], "
                  f"total={loss_stats['total_loss_updates']}")
