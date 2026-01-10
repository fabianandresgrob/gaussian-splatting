"""
Scheduled hybrid view selection strategy.

This selector combines multiple selectors with time-dependent weights,
enabling adaptive strategies that evolve during training.
"""

import numpy as np
from typing import Dict, List, Tuple
from .selector import ViewSelector


# Standard configurations for common training strategies
STANDARD_CONFIGS = {
    'explore_then_exploit': {
        'description': 'Start with diversity (clustering), transition to difficulty (loss-based)',
        'selectors': ['clustering', 'loss_based'],
        'weights_start': [0.8, 0.2],
        'weights_end': [0.2, 0.8],
        'schedule_type': 'linear',
        'max_iterations': 30000,
        'temperature': 1.0,
    },

    'geometric_to_gaussian': {
        'description': 'Start with geometric heuristics, transition to Gaussian-aware',
        'selectors': ['geometric', 'gaussian_aware'],
        'weights_start': [0.9, 0.1],
        'weights_end': [0.3, 0.7],
        'schedule_type': 'cosine',
        'max_iterations': 30000,
        'temperature': 1.0,
        'gaussian_aware_config': {
            'mode': 'inverse_density',
            'update_frequency': 500,
        },
    },

    'three_phase_training': {
        'description': 'Three-phase: diversity → balance → loss-focused',
        'selectors': ['clustering', 'geometric', 'loss_based'],
        'schedule_type': 'step',
        'milestones': [
            [0,     [0.6, 0.3, 0.1]],   # Phase 1: Diversity (clustering)
            [10000, [0.3, 0.4, 0.3]],   # Phase 2: Balanced
            [20000, [0.1, 0.2, 0.7]],   # Phase 3: Loss-focused
        ],
        'temperature': 1.0,
    },
}


class ScheduledHybridSelector(ViewSelector):
    """
    Selector that combines multiple selectors with time-dependent weights.

    Supports three schedule types:
    - 'linear': Linear interpolation between start and end weights
    - 'cosine': Cosine annealing schedule
    - 'step': Piecewise constant weights at defined milestones

    This enables adaptive strategies like:
    - Start with diversity (clustering) → transition to difficulty (loss-based)
    - Start with geometric heuristics → transition to Gaussian-aware
    - Multi-phase training with different priorities
    """

    def __init__(self, config: dict = None, log_dir: str = None, verbose: bool = False, seed: int = None):
        """
        Initialize the scheduled hybrid selector.

        Args:
            config: Configuration dictionary with required keys:
                - selectors (list): List of selector names to combine
                - schedule_type (str): 'linear', 'cosine', or 'step'

                For 'linear' and 'cosine':
                - weights_start (list): Initial weights (sum to 1.0)
                - weights_end (list): Final weights (sum to 1.0)
                - max_iterations (int): Iteration at which to reach weights_end

                For 'step':
                - milestones (list): List of [iteration, weights] pairs

                Optional:
                - temperature (float): Softmax temperature. Default: 1.0
                - <selector_name>_config (dict): Config for specific sub-selector

                Or use a standard config:
                - preset (str): One of 'explore_then_exploit', 'geometric_to_gaussian',
                  'three_phase_training'

            log_dir: Directory to save selection logs
            verbose: If True, print detailed information
            seed: Random seed for reproducibility
        """
        super().__init__(config, log_dir, verbose, seed)

        # Local import to avoid circular import with view_selection/__init__.py
        from . import build_selector

        # Check for preset configuration
        if 'preset' in self.config:
            preset_name = self.config['preset']
            if preset_name not in STANDARD_CONFIGS:
                raise ValueError(
                    f"Unknown preset '{preset_name}'. "
                    f"Available: {list(STANDARD_CONFIGS.keys())}"
                )
            # Merge preset with user config (user config overrides preset)
            preset_config = STANDARD_CONFIGS[preset_name].copy()
            preset_config.update(self.config)
            self.config = preset_config

        # Validate required config
        if 'selectors' not in self.config:
            raise ValueError("Config must specify 'selectors' list")

        if 'schedule_type' not in self.config:
            raise ValueError("Config must specify 'schedule_type'")

        self.selector_names = self.config['selectors']
        self.schedule_type = self.config['schedule_type']
        self.temperature = self.config.get('temperature', 1.0)

        # Schedule-specific config
        if self.schedule_type in ['linear', 'cosine']:
            if 'weights_start' not in self.config or 'weights_end' not in self.config:
                raise ValueError(f"{self.schedule_type} schedule requires 'weights_start' and 'weights_end'")
            if 'max_iterations' not in self.config:
                raise ValueError(f"{self.schedule_type} schedule requires 'max_iterations'")

            self.weights_start = self.config['weights_start']
            self.weights_end = self.config['weights_end']
            self.max_iterations = self.config['max_iterations']

            if len(self.weights_start) != len(self.selector_names):
                raise ValueError("weights_start length must match selectors length")
            if len(self.weights_end) != len(self.selector_names):
                raise ValueError("weights_end length must match selectors length")

        elif self.schedule_type == 'step':
            if 'milestones' not in self.config:
                raise ValueError("'step' schedule requires 'milestones'")

            self.milestones = self.config['milestones']
            # Sort milestones by iteration
            self.milestones.sort(key=lambda x: x[0])

            # Validate milestone weights
            for iteration, weights in self.milestones:
                if len(weights) != len(self.selector_names):
                    raise ValueError(f"Milestone at {iteration}: weights length must match selectors length")

        else:
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")

        # Create sub-selectors
        self.sub_selectors = []
        source_path = self.config.get("source_path", None)
        for name in self.selector_names:
            # Get selector-specific config if provided
            sub_config_key = f'{name}_config'
            sub_config = self.config.get(sub_config_key, {})

            # Ensure sub-selector sees dataset root for path resolution (e.g., DINO embeddings).
            if source_path and isinstance(sub_config, dict):
                sub_config.setdefault("source_path", source_path)

            # Create selector
            selector = build_selector(
                name,
                config=sub_config,
                log_dir=None,  # Don't duplicate logging
                verbose=self.verbose,
                seed=seed
            )
            self.sub_selectors.append(selector)

        if self.verbose:
            self.logger.debug(f"Configuration: selectors={self.selector_names}, "
                            f"schedule={self.schedule_type}, temp={self.temperature}")
            if self.schedule_type in ['linear', 'cosine']:
                self.logger.debug(f"Weights: {self.weights_start} → {self.weights_end}, "
                                 f"max_iter={self.max_iterations}")
            elif self.schedule_type == 'step':
                self.logger.debug(f"Milestones: {len(self.milestones)} phases")

    def initialize(self, all_cameras: List) -> None:
        """
        Initialize all sub-selectors.

        Args:
            all_cameras: List of all available Camera objects
        """
        super().initialize(all_cameras)

        # Initialize each sub-selector
        for selector in self.sub_selectors:
            selector.initialize(all_cameras)

        self.initialized = True

        self.logger.info(f"Initialized {len(self.sub_selectors)} sub-selectors: {self.selector_names}")

    def _get_current_weights(self, iteration: int) -> List[float]:
        """
        Compute current selector weights based on schedule.

        Args:
            iteration: Current training iteration

        Returns:
            List of weights for each selector
        """
        if self.schedule_type == 'linear':
            # Linear interpolation
            t = min(iteration / self.max_iterations, 1.0)
            weights = []
            for start, end in zip(self.weights_start, self.weights_end):
                weight = start + (end - start) * t
                weights.append(weight)

        elif self.schedule_type == 'cosine':
            # Cosine annealing: smooth transition
            t = min(iteration / self.max_iterations, 1.0)
            # Cosine from 1.0 (start) to 0.0 (end)
            cosine_factor = 0.5 * (1.0 + np.cos(np.pi * t))

            weights = []
            for start, end in zip(self.weights_start, self.weights_end):
                # Interpolate: end + (start - end) * cosine_factor
                weight = end + (start - end) * cosine_factor
                weights.append(weight)

        elif self.schedule_type == 'step':
            # Piecewise constant weights
            # Find the last milestone that iteration has reached
            current_weights = self.milestones[0][1]  # Default to first
            for milestone_iter, milestone_weights in self.milestones:
                if iteration >= milestone_iter:
                    current_weights = milestone_weights
                # No break - we want to find the LAST applicable milestone
            weights = current_weights

        else:
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")

        # Normalize to ensure sum = 1.0
        total = sum(weights)
        weights = [w / total for w in weights]

        return weights

    def compute_probabilities(self, gaussians, iteration: int) -> Dict[int, float]:
        """
        Combine probabilities from all sub-selectors with current weights.

        Args:
            gaussians: Current Gaussian model
            iteration: Current training iteration

        Returns:
            Dictionary mapping camera uid to probability
        """
        # Get current weights
        weights = self._get_current_weights(iteration)

        # Initialize combined scores
        combined_scores = {cam.uid: 0.0 for cam in self.all_cameras}

        # Combine probabilities from each selector
        for selector, weight in zip(self.sub_selectors, weights):
            probs = selector.compute_probabilities(gaussians, iteration)

            # Add weighted probabilities
            for uid, prob in probs.items():
                combined_scores[uid] += weight * prob

        # Renormalize (should already be normalized, but ensure it)
        total = sum(combined_scores.values())
        if total > 0:
            combined_scores = {uid: score / total for uid, score in combined_scores.items()}

        # Optional: apply hybrid-level temperature to sharpen/flatten the combined distribution.
        # We interpret temperature as acting on probabilities via p^(1/T), which has the
        # intuitive behavior: T<1 => peakier, T>1 => flatter.
        if self.temperature is not None and float(self.temperature) > 0 and float(self.temperature) != 1.0:
            eps = 1e-12
            t = float(self.temperature)
            uids = list(combined_scores.keys())
            p = np.array([combined_scores[uid] for uid in uids], dtype=np.float64)
            p = np.power(np.clip(p, eps, 1.0), 1.0 / t)
            p = p / p.sum()
            combined_scores = {uid: float(p[i]) for i, uid in enumerate(uids)}

        # Log current weights periodically
        if self.verbose and iteration % 1000 == 0 and iteration > 0:
            self.logger.debug(f"Iter {iteration}: weights={[f'{w:.3f}' for w in weights]} "
                            f"selectors={self.selector_names}")

        return combined_scores

    def select_view(self, gaussians, iteration: int):
        """Select a camera using the combined distribution and notify sub-selectors.

        This override is important for *stateful* sub-selectors whose probabilities
        depend on the history of selections (e.g. DINO 'distance_to_selected',
        clustering selection-count adaptation). If we only call their
        compute_probabilities() but never forward which camera was actually chosen,
        their internal state would never update and their behavior would silently
        differ from the intended design.
        """
        if not self.initialized:
            raise RuntimeError("ViewSelector must be initialized before use. Call initialize() first.")

        # Get current weights
        weights = self._get_current_weights(iteration)

        # Compute each sub-selector's probabilities (keep for notification)
        uids = [cam.uid for cam in self.all_cameras]
        uid_to_idx = {uid: i for i, uid in enumerate(uids)}

        combined = np.zeros(len(uids), dtype=np.float64)
        per_selector_probs: List[Dict[int, float]] = []

        for selector, weight in zip(self.sub_selectors, weights):
            probs = selector.compute_probabilities(gaussians, iteration)
            per_selector_probs.append(probs)
            for uid, prob in probs.items():
                idx = uid_to_idx.get(uid, None)
                if idx is not None:
                    combined[idx] += float(weight) * float(prob)

        # Normalize combined distribution
        s = float(combined.sum())
        if s <= 0:
            combined = np.ones_like(combined) / float(len(combined))
        else:
            combined = combined / s

        # Apply hybrid-level temperature (same semantics as compute_probabilities)
        if self.temperature is not None and float(self.temperature) > 0 and float(self.temperature) != 1.0:
            eps = 1e-12
            t = float(self.temperature)
            combined = np.power(np.clip(combined, eps, 1.0), 1.0 / t)
            combined = combined / float(combined.sum())

        # Sample
        selected_idx = int(self.rng.choice(len(uids), p=combined))
        selected_uid = uids[selected_idx]
        selected_cam = None
        for cam in self.all_cameras:
            if cam.uid == selected_uid:
                selected_cam = cam
                break
        if selected_cam is None:
            raise RuntimeError(f"Camera with uid {selected_uid} not found in all_cameras")

        # Log hybrid selection
        self.log_selection(selected_cam, float(combined[selected_idx]), iteration)

        # Notify sub-selectors of the realized selection so stateful ones update.
        for selector, probs in zip(self.sub_selectors, per_selector_probs):
            if hasattr(selector, "log_selection"):
                selector.log_selection(selected_cam, float(probs.get(selected_uid, 0.0)), iteration)

        return selected_cam

    def update_loss(self, camera, loss: float) -> None:
        """
        Forward loss updates to sub-selectors that support it.

        Args:
            camera: Camera object that was just rendered
            loss: Loss value from rendering
        """
        for selector in self.sub_selectors:
            if hasattr(selector, 'update_loss'):
                selector.update_loss(camera, loss)

    def update_coverage_counts(self, gaussians, camera=None) -> None:
        """
        Forward coverage updates to sub-selectors that support it.

        Args:
            gaussians: Current Gaussian model
            camera: Optional camera to update
        """
        for selector in self.sub_selectors:
            if hasattr(selector, 'update_coverage_counts'):
                selector.update_coverage_counts(gaussians, camera)

    def get_hybrid_statistics(self, iteration: int) -> Dict:
        """
        Get statistics about the hybrid selector and current weights.

        Args:
            iteration: Current training iteration

        Returns:
            Dictionary with hybrid selector statistics
        """
        weights = self._get_current_weights(iteration)

        stats = {
            'selectors': self.selector_names,
            'current_weights': {
                name: float(weight)
                for name, weight in zip(self.selector_names, weights)
            },
            'schedule_type': self.schedule_type,
            'current_iteration': iteration,
        }

        # Add schedule-specific info
        if self.schedule_type in ['linear', 'cosine']:
            stats['max_iterations'] = self.max_iterations
            stats['progress'] = min(iteration / self.max_iterations, 1.0)
        elif self.schedule_type == 'step':
            # Find current phase
            current_phase = 0
            for i, (milestone_iter, _) in enumerate(self.milestones):
                if iteration >= milestone_iter:
                    current_phase = i
            stats['current_phase'] = current_phase
            stats['total_phases'] = len(self.milestones)

        return stats

    def log_statistics(self, iteration: int) -> None:
        """
        Print hybrid selector and sub-selector statistics.

        Args:
            iteration: Current training iteration
        """
        # Get base selection statistics
        selection_stats = self.get_selection_statistics()

        # Get hybrid statistics
        hybrid_stats = self.get_hybrid_statistics(iteration)

        self.logger.info(f"Statistics at iteration {iteration}:")
        self.logger.info(f"  Schedule: {self.schedule_type}, selectors: {self.selector_names}")
        
        weight_str = ', '.join(f"{name}={w:.3f}" for name, w in hybrid_stats['current_weights'].items())
        self.logger.info(f"  Weights: {weight_str}")

        if self.schedule_type in ['linear', 'cosine']:
            self.logger.info(f"  Progress: {hybrid_stats['progress']:.1%}")
        elif self.schedule_type == 'step':
            self.logger.info(f"  Phase: {hybrid_stats['current_phase'] + 1}/{hybrid_stats['total_phases']}")

        self.logger.info(f"  Selections: {selection_stats.get('total_selections', 0)}, "
                        f"unique cameras: {selection_stats.get('unique_cameras', 0)}")


def get_standard_config(preset_name: str) -> dict:
    """
    Get a standard configuration by name.

    Args:
        preset_name: Name of the preset configuration

    Returns:
        Configuration dictionary

    Available presets:
    - 'explore_then_exploit': Start with diversity, transition to difficulty
    - 'geometric_to_gaussian': Start with geometric heuristics, transition to Gaussian-aware
    - 'three_phase_training': Three-phase training with different priorities
    """
    if preset_name not in STANDARD_CONFIGS:
        raise ValueError(
            f"Unknown preset '{preset_name}'. "
            f"Available: {list(STANDARD_CONFIGS.keys())}"
        )
    return STANDARD_CONFIGS[preset_name].copy()
