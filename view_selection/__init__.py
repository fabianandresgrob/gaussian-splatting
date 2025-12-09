"""
View selection module for intelligent camera sampling in 3D Gaussian Splatting.

This module provides different strategies for selecting which camera viewpoint
to render from during training, replacing naive random selection with more
sophisticated approaches.

Available strategies:
- RandomSelector: Uniform random selection (baseline)
- FixedProbabilitySelector: Fixed probabilities based on pose heuristics
- EpochBasedSelector: Dynamic probabilities with recency penalty
- ClusteringSelector: Cluster-based selection for diversity
- WithoutReplacementSelector: Baseline random sampling without Replacement
- LossBasedSelector: Loss-driven selection prioritizing harder views
- GaussianAwareSelector: Adaptive selection based on Gaussian visibility
- ScheduledHybridSelector: Combines multiple selectors with time-dependent weights
"""

from .selector import ViewSelector
from .random_selector import RandomSelector
from .heuristic_selector import FixedProbabilitySelector
from .epoch_selector import EpochBasedSelector
from .clustering_selector import ClusteringSelector
from .no_replace_selector import WithoutReplacementSelector
from .loss_selector import LossBasedSelector
from .gaussian_aware_selector import GaussianAwareSelector
from .scheduled_hybrid_selector import ScheduledHybridSelector, get_standard_config, STANDARD_CONFIGS


# Registry mapping strategy names to classes
SELECTOR_REGISTRY = {
    'random': RandomSelector,
    'fixed_prob': FixedProbabilitySelector,
    'epoch_based': EpochBasedSelector,
    'clustering': ClusteringSelector,
    'no_replace': WithoutReplacementSelector,
    'loss_based': LossBasedSelector,
    'gaussian_aware': GaussianAwareSelector,
    'scheduled_hybrid': ScheduledHybridSelector,
}


def build_selector(selector_type: str, config: dict = None, log_dir: str = None, verbose: bool = False, seed: int = None) -> ViewSelector:
    """
    Factory function to create a view selector.

    Args:
        selector_type: Name of the selector strategy. Must be one of:
            - 'random': Uniform random selection
            - 'fixed_prob': Fixed probabilities based on pose heuristics
            - 'epoch_based': Dynamic probabilities with recency penalty
            - 'clustering': Cluster-based selection
            - 'no_replace': Random sampling without replacement
            - 'loss_based': Loss-driven selection prioritizing harder views
            - 'gaussian_aware': Adaptive selection based on Gaussian visibility
            - 'scheduled_hybrid': Combines multiple selectors with time-dependent weights
        config: Configuration dictionary for the selector (strategy-specific)
        log_dir: Directory to save selection logs
        verbose: If True, print detailed information
        seed: Random seed for reproducibility

    Returns:
        Initialized ViewSelector instance

    Raises:
        ValueError: If selector_type is not recognized

    Examples:
        >>> # Create a random selector
        >>> selector = build_selector('random')

        >>> # Create a clustering selector with custom config
        >>> config = {'n_clusters': 15, 'temperature': 0.8}
        >>> selector = build_selector('clustering', config=config, verbose=True, seed=42)

        >>> # Create a loss-based selector
        >>> config = {'ema_decay': 0.99, 'temperature': 1.0, 'min_samples_before_bias': 5}
        >>> selector = build_selector('loss_based', config=config, verbose=True)

        >>> # Create a Gaussian-aware selector
        >>> config = {'mode': 'inverse_density', 'update_frequency': 500}
        >>> selector = build_selector('gaussian_aware', config=config, verbose=True)

        >>> # Create a scheduled hybrid selector with preset
        >>> config = {'preset': 'explore_then_exploit'}
        >>> selector = build_selector('scheduled_hybrid', config=config, verbose=True)
    """
    if selector_type not in SELECTOR_REGISTRY:
        available = ', '.join(SELECTOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown selector type '{selector_type}'. "
            f"Available selectors: {available}"
        )

    selector_class = SELECTOR_REGISTRY[selector_type]
    return selector_class(config=config, log_dir=log_dir, verbose=verbose, seed=seed)


def list_selectors():
    """
    Get a list of available selector types.

    Returns:
        List of selector type names
    """
    return list(SELECTOR_REGISTRY.keys())


__all__ = [
    'ViewSelector',
    'RandomSelector',
    'FixedProbabilitySelector',
    'EpochBasedSelector',
    'ClusteringSelector',
    'WithoutReplacementSelector',
    'LossBasedSelector',
    'GaussianAwareSelector',
    'ScheduledHybridSelector',
    'build_selector',
    'list_selectors',
    'get_standard_config',
    'STANDARD_CONFIGS',
    'SELECTOR_REGISTRY',
]
