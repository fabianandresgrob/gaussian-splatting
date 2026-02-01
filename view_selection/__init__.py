"""View selection module for intelligent camera sampling in 3D Gaussian Splatting.

This module provides different strategies for selecting which camera viewpoint
to render from during training, replacing naive uniform sampling with more
sophisticated approaches.

Available strategies:
- StackBasedSelector: Stack-based shuffle (default 3DGS baseline) - guaranteed uniform coverage
- UniformRandomSelector: True uniform random with replacement
- GeometricDiversitySelector: Pose-based heuristics for geometric diversity
- ClusteringSelector: Cluster-based selection for spatial coverage
- LossBasedSelector: Loss-driven selection prioritizing harder views
- DINOSelector: DINO feature-based diversity sampling
- SequentialSelector: Deterministic sequential iteration
- DeterministicMaxLossSelector: Always select highest-loss camera
"""

from .selector import ViewSelector
from .random_selector import UniformRandomSelector
from .geometric_selector import GeometricDiversitySelector
from .clustering_selector import ClusteringSelector
from .stack_selector import StackBasedSelector
from .loss_selector import LossBasedSelector
from .sequential_selector import SequentialSelector
from .deterministic_loss_selector import DeterministicMaxLossSelector
from .logging_utils import configure_logging, get_logger

# Optional selectors (may require extra deps)
try:
    from .dino_selector import DINOSelector
except Exception:  # pragma: no cover
    DINOSelector = None

# Registry mapping strategy names to classes
SELECTOR_REGISTRY = {
    'stack': StackBasedSelector,
    'uniform_random': UniformRandomSelector,
    'geometric': GeometricDiversitySelector,
    'clustering': ClusteringSelector,
    'loss_based': LossBasedSelector,
    'sequential': SequentialSelector,
    'deterministic_max_loss': DeterministicMaxLossSelector,
}

if DINOSelector is not None:
    SELECTOR_REGISTRY['dino'] = DINOSelector


def build_selector(selector_type: str, config: dict = None, log_dir: str = None, verbose: bool = False, seed: int = None) -> ViewSelector:
    """
    Factory function to create a view selector.

    Args:
        selector_type: Name of the selector strategy. Options:
            - 'stack': Stack-based shuffle, default 3DGS baseline
            - 'uniform_random': True random with replacement
            - 'geometric': Geometric diversity from poses
            - 'clustering': Cluster-based selection
            - 'loss_based': Loss-driven selection
            - 'dino': DINO feature-based diversity
            - 'sequential': Deterministic sequential iteration
            - 'deterministic_max_loss': Always select highest-loss camera

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
    # Selectors
    'StackBasedSelector',
    'UniformRandomSelector',
    'GeometricDiversitySelector',
    'ClusteringSelector',
    'LossBasedSelector',
    'SequentialSelector',
    'DeterministicMaxLossSelector',
    # Utilities
    'build_selector',
    'list_selectors',
    'SELECTOR_REGISTRY',
    # Logging
    'configure_logging',
    'get_logger',
]

if DINOSelector is not None:
    __all__.append('DINOSelector')
