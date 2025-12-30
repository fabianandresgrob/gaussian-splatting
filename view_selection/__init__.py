"""
View selection module for intelligent camera sampling in 3D Gaussian Splatting.

This module provides different strategies for selecting which camera viewpoint
to render from during training, replacing naive random selection with more
sophisticated approaches.

Available strategies (matching slides terminology):
- StackBasedSelector: Stack-based shuffle (default 3DGS baseline) - guaranteed uniform coverage
- UniformRandomSelector: True random with replacement (baseline for probability methods)
- GeometricDiversitySelector: Pose-based heuristics for geometric diversity
- ClusteringSelector: Cluster-based selection for spatial coverage
- LossBasedSelector: Loss-driven selection prioritizing harder views
- GaussianAwareSelector: Adaptive selection based on Gaussian visibility
- ScheduledHybridSelector: Combines multiple selectors with time-dependent weights
- VGGTSelector: VGGT-guided selection (optional advanced method)

Legacy aliases maintained for backward compatibility:
- RandomSelector -> UniformRandomSelector
- FixedProbabilitySelector -> GeometricDiversitySelector
- WithoutReplacementSelector -> StackBasedSelector
"""

from .selector import ViewSelector
from .random_selector import UniformRandomSelector
from .geometric_selector import GeometricDiversitySelector
from .clustering_selector import ClusteringSelector
from .stack_selector import StackBasedSelector
from .loss_selector import LossBasedSelector
from .gaussian_aware_selector import GaussianAwareSelector
from .scheduled_hybrid_selector import ScheduledHybridSelector, get_standard_config, STANDARD_CONFIGS
from .dino_selector import DINOSelector
from .vggt_selector import VGGTSelector

# Legacy aliases for backward compatibility
RandomSelector = UniformRandomSelector
FixedProbabilitySelector = GeometricDiversitySelector
WithoutReplacementSelector = StackBasedSelector


# Registry mapping strategy names to classes
SELECTOR_REGISTRY = {
    # New names (matching slides)
    'stack': StackBasedSelector,
    'uniform_random': UniformRandomSelector,
    'geometric': GeometricDiversitySelector,
    'clustering': ClusteringSelector,
    'loss_based': LossBasedSelector,
    'gaussian_aware': GaussianAwareSelector,
    'scheduled_hybrid': ScheduledHybridSelector,
    'dino': DINOSelector,
    'vggt': VGGTSelector,
    # Legacy aliases for backward compatibility
    'random': UniformRandomSelector,
    'fixed_prob': GeometricDiversitySelector,
    'no_replace': StackBasedSelector,
}


def build_selector(selector_type: str, config: dict = None, log_dir: str = None, verbose: bool = False, seed: int = None) -> ViewSelector:
    """
    Factory function to create a view selector.

    Args:
        selector_type: Name of the selector strategy. Options:
            New names (recommended):
            - 'stack': Stack-based shuffle, default 3DGS baseline
            - 'uniform_random': True random with replacement
            - 'geometric': Geometric diversity from poses
            - 'clustering': Cluster-based selection
            - 'loss_based': Loss-driven selection
            - 'gaussian_aware': Adaptive selection based on Gaussian visibility
            - 'scheduled_hybrid': Combines multiple selectors
            - 'vggt': VGGT-guided selection
            
            Legacy names (still supported):
            - 'random': Alias for 'uniform_random'
            - 'fixed_prob': Alias for 'geometric'
            - 'no_replace': Alias for 'stack'
            
        config: Configuration dictionary for the selector (strategy-specific)
        log_dir: Directory to save selection logs
        verbose: If True, print detailed information
        seed: Random seed for reproducibility

    Returns:
        Initialized ViewSelector instance

    Raises:
        ValueError: If selector_type is not recognized

    Examples:
        >>> # Create a stack-based selector (default baseline)
        >>> selector = build_selector('stack')

        >>> # Create a geometric diversity selector
        >>> config = {'temperature': 0.8, 'distance_weight': 0.6}
        >>> selector = build_selector('geometric', config=config, verbose=True)

        >>> # Create a clustering selector
        >>> config = {'n_clusters': 15, 'temperature': 0.8}
        >>> selector = build_selector('clustering', config=config, verbose=True, seed=42)
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
    # Base class
    'ViewSelector',
    # New names (primary)
    'StackBasedSelector',
    'UniformRandomSelector',
    'GeometricDiversitySelector',
    'ClusteringSelector',
    'LossBasedSelector',
    'GaussianAwareSelector',
    'ScheduledHybridSelector',
    'DINOSelector',
    'VGGTSelector',
    # Legacy aliases
    'RandomSelector',
    'FixedProbabilitySelector',
    'WithoutReplacementSelector',
    # Utilities
    'build_selector',
    'list_selectors',
    'get_standard_config',
    'STANDARD_CONFIGS',
    'SELECTOR_REGISTRY',
]
