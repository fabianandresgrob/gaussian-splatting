"""
Unit tests for view selection strategies.

Tests all selector implementations for correct behavior, probability distributions,
and reproducibility.
"""

import pytest
import numpy as np
import torch
from typing import List
from unittest.mock import Mock

# Import view selection components
import sys
sys.path.insert(0, '..')
from view_selection import build_selector, list_selectors, SELECTOR_REGISTRY

# Selectors that are deterministic, not fully implemented, or require special setup
# - sequential: Deterministic, assigns prob 1.0 to current camera only
# - deterministic_max_loss: Requires render context
# - gaussian_aware: Not fully implemented/used
# - scheduled_hybrid: Not fully implemented/used
# - vggt: Requires external model
SPECIAL_SELECTORS = {'sequential', 'deterministic_max_loss', 'gaussian_aware', 'scheduled_hybrid', 'vggt'}


class MockCamera:
    """Mock camera object for testing view selectors."""

    def __init__(self, uid: int, position: np.ndarray, rotation: np.ndarray = None, image_name: str = None):
        """
        Create a mock camera.

        Args:
            uid: Unique camera identifier
            position: 3D position [x, y, z]
            rotation: 3x3 rotation matrix (optional, identity if not provided)
            image_name: Image filename (optional, auto-generated if not provided)
        """
        self.uid = uid
        self.image_name = image_name or f"image_{uid:04d}.jpg"

        # Camera position as torch tensor
        self.camera_center = torch.tensor(position, dtype=torch.float32)

        # Rotation matrix (world-to-camera)
        if rotation is None:
            # Identity rotation by default
            self.R = np.eye(3, dtype=np.float32)
        else:
            self.R = rotation.astype(np.float32)


@pytest.fixture
def mock_cameras_simple():
    """Create a simple set of 10 mock cameras arranged in a circle."""
    cameras = []
    n_cameras = 10
    radius = 5.0

    for i in range(n_cameras):
        angle = 2 * np.pi * i / n_cameras
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 1.0  # Fixed height

        # Rotation matrix looking toward center
        # Simple rotation around Z axis
        cos_a, sin_a = np.cos(angle + np.pi), np.sin(angle + np.pi)
        R = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        camera = MockCamera(uid=i, position=np.array([x, y, z]), rotation=R)
        cameras.append(camera)

    return cameras


@pytest.fixture
def mock_cameras_diverse():
    """Create a diverse set of 20 mock cameras with varying positions and orientations."""
    cameras = []
    n_cameras = 20

    # Create cameras at random positions
    rng = np.random.RandomState(42)

    for i in range(n_cameras):
        # Random position in a cube
        position = rng.uniform(-10, 10, size=3)

        # Random rotation matrix (using Euler angles)
        angles = rng.uniform(0, 2*np.pi, size=3)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])]
        ])
        Ry = np.array([
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])]
        ])
        Rz = np.array([
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1]
        ])
        R = Rz @ Ry @ Rx

        camera = MockCamera(uid=i, position=position, rotation=R)
        cameras.append(camera)

    return cameras


@pytest.fixture
def mock_gaussians():
    """Create a mock Gaussian model (most selectors don't use this)."""
    gaussians = Mock()
    return gaussians


# Test 1: All selectors can be created via build_selector()
def test_selector_registry(mock_cameras_simple):
    """Test that all registered selectors can be instantiated."""
    available_selectors = list_selectors()

    assert len(available_selectors) > 0, "No selectors registered"

    for selector_name in available_selectors:
        # Skip selectors that require special config
        if selector_name == 'scheduled_hybrid':
            config = {
                'selectors': ['uniform_random', 'loss_based'],
                'weights_start': [0.5, 0.5],
                'weights_end': [0.5, 0.5],
                'schedule_type': 'linear',
                'max_iterations': 1000,
            }
            selector = build_selector(selector_name, config=config, seed=42)
        elif selector_name == 'vggt':
            # VGGT requires external model, skip in registry test
            print(f"~ {selector_name} skipped (requires external model)")
            continue
        else:
            selector = build_selector(selector_name, seed=42)

        # Initialize with cameras
        selector.initialize(mock_cameras_simple)

        assert selector.initialized, f"Selector '{selector_name}' failed to initialize"
        print(f"✓ {selector_name} initialized successfully")


# Test 2: compute_probabilities() sums to 1.0
@pytest.mark.parametrize("selector_name", [s for s in list_selectors() if s not in SPECIAL_SELECTORS])
def test_probability_sum(selector_name, mock_cameras_simple, mock_gaussians):
    """Test that probabilities sum to 1.0 for each selector.
    
    Note: Some selectors are excluded:
    - deterministic_max_loss: Requires render context
    - gaussian_aware: Requires real Gaussian model
    - scheduled_hybrid: Requires config with sub-selectors
    - sequential: Deterministic (only one camera has prob 1.0)
    """
    selector = build_selector(selector_name, seed=42)
    selector.initialize(mock_cameras_simple)

    # Compute probabilities at different iterations
    for iteration in [0, 100, 500, 1000]:
        probs = selector.compute_probabilities(mock_gaussians, iteration)

        prob_sum = sum(probs.values())

        assert abs(prob_sum - 1.0) < 1e-6, (
            f"Selector '{selector_name}' probabilities sum to {prob_sum}, not 1.0 "
            f"at iteration {iteration}"
        )


# Test 3: Reproducibility with same seed
@pytest.mark.parametrize("selector_name", [s for s in list_selectors() if s not in SPECIAL_SELECTORS])
def test_reproducibility(selector_name, mock_cameras_simple, mock_gaussians):
    """Test that same seed produces same selection sequence.
    
    Note: Some selectors are excluded:
    - deterministic_max_loss: Requires render context
    - gaussian_aware: Requires real Gaussian model
    - scheduled_hybrid: Requires config with sub-selectors
    """
    n_selections = 20

    # First run
    selector1 = build_selector(selector_name, seed=42)
    selector1.initialize(mock_cameras_simple)

    selections1 = []
    for i in range(n_selections):
        cam = selector1.select_view(mock_gaussians, iteration=i)
        selections1.append(cam.uid)

    # Second run with same seed
    selector2 = build_selector(selector_name, seed=42)
    selector2.initialize(mock_cameras_simple)

    selections2 = []
    for i in range(n_selections):
        cam = selector2.select_view(mock_gaussians, iteration=i)
        selections2.append(cam.uid)

    assert selections1 == selections2, (
        f"Selector '{selector_name}' not reproducible with same seed. "
        f"First run: {selections1[:5]}..., Second run: {selections2[:5]}..."
    )


# Test 4: All cameras have non-zero probability
@pytest.mark.parametrize("selector_name", [s for s in list_selectors() if s not in SPECIAL_SELECTORS])
def test_all_cameras_nonzero_prob(selector_name, mock_cameras_simple, mock_gaussians):
    """Test that all cameras have non-zero sampling probability.
    
    Note: Some selectors are excluded:
    - sequential: Deterministic, assigns prob 1.0 to current camera only
    - deterministic_max_loss: Requires render context
    - gaussian_aware: Not fully implemented/used
    - scheduled_hybrid: Not fully implemented/used
    """
    selector = build_selector(selector_name, seed=42)
    selector.initialize(mock_cameras_simple)

    probs = selector.compute_probabilities(mock_gaussians, iteration=0)

    zero_prob_cameras = [uid for uid, prob in probs.items() if prob == 0.0]

    assert len(zero_prob_cameras) == 0, (
        f"Selector '{selector_name}' assigned zero probability to cameras: {zero_prob_cameras}"
    )

    # Check minimum probability is reasonable (not too small)
    min_prob = min(probs.values())
    expected_min_reasonable = 1e-4  # At least 0.01% chance

    assert min_prob >= expected_min_reasonable, (
        f"Selector '{selector_name}' has very low minimum probability: {min_prob}"
    )


# Test 5: ClusteringSelector assigns all cameras to clusters
@pytest.mark.parametrize("clustering_method", ['kmeans', 'dbscan'])
def test_clustering_all_assigned(clustering_method, mock_cameras_diverse, mock_gaussians):
    """Test that ClusteringSelector assigns all cameras to a cluster."""
    # Config for clustering
    if clustering_method == 'kmeans':
        config = {
            'clustering_method': 'kmeans',
            'n_clusters': 5,
            'use_orientation': True
        }
    else:  # dbscan
        config = {
            'clustering_method': 'dbscan',
            'eps': 2.0,  # Larger eps to ensure cameras are clustered
            'min_samples': 2,
            'use_orientation': True
        }

    selector = build_selector('clustering', config=config, seed=42)
    selector.initialize(mock_cameras_diverse)

    # Check that all cameras are assigned to a cluster
    n_cameras = len(mock_cameras_diverse)
    assert len(selector.camera_clusters) == n_cameras, (
        f"Not all cameras assigned to clusters. "
        f"Expected {n_cameras}, got {len(selector.camera_clusters)}"
    )

    # Check that all cameras have valid cluster IDs
    for cam in mock_cameras_diverse:
        assert cam.uid in selector.camera_clusters, f"Camera {cam.uid} not assigned to cluster"
        cluster_id = selector.camera_clusters[cam.uid]
        assert cluster_id in selector.cluster_sizes, f"Cluster {cluster_id} not in cluster_sizes"

    # Get cluster statistics
    stats = selector.get_cluster_statistics()
    print(f"✓ {clustering_method}: {stats['n_clusters']} clusters, "
          f"sizes: {list(stats['cluster_sizes'].values())}")


# Test 6: LossBasedSelector updates loss correctly
def test_loss_based_selector_updates(mock_cameras_simple, mock_gaussians):
    """Test that LossBasedSelector correctly updates losses."""
    config = {
        'temperature': 1.0,
        'min_samples_before_bias': 3
    }
    selector = build_selector('loss_based', config=config, seed=42)
    selector.initialize(mock_cameras_simple)

    # Initially all cameras should have zero loss
    assert all(loss == 0.0 for loss in selector.losses.values()), \
        "Initial losses should be zero"

    # Simulate training: select cameras and update losses
    for i in range(30):
        cam = selector.select_view(mock_gaussians, iteration=i)

        # Simulate different loss values
        # Camera 0 has high loss, others have low loss
        loss_value = 1.0 if cam.uid == 0 else 0.1
        selector.update_loss(cam, loss_value)

    # Check that sample counts increased
    total_samples = sum(selector.loss_sample_counts.values())
    assert total_samples == 30, f"Expected 30 loss updates, got {total_samples}"

    print(f"✓ LossBasedSelector updated losses correctly")
    print(f"  Losses: {dict(list(selector.losses.items())[:3])}")


# Test 7: Clustering with DBSCAN handles noise points
def test_dbscan_noise_handling(mock_cameras_diverse, mock_gaussians):
    """Test that DBSCAN correctly handles noise points (outliers)."""
    # Use strict DBSCAN parameters to force some noise points
    config = {
        'clustering_method': 'dbscan',
        'eps': 0.5,  # Small eps
        'min_samples': 5,  # High min_samples
        'use_orientation': True
    }

    selector = build_selector('clustering', config=config, seed=42)
    selector.initialize(mock_cameras_diverse)

    # All cameras should still be assigned (including outliers)
    n_cameras = len(mock_cameras_diverse)
    assert len(selector.camera_clusters) == n_cameras, \
        "All cameras should be assigned even with noise points"

    # Check if there's an outlier cluster (cluster with id >= n_regular_clusters)
    cluster_ids = list(selector.cluster_sizes.keys())
    print(f"✓ DBSCAN with strict params: {len(cluster_ids)} total clusters")
    print(f"  Cluster sizes: {selector.cluster_sizes}")


# Test 8: Selection statistics tracking
def test_selection_statistics(mock_cameras_simple, mock_gaussians):
    """Test that selectors correctly track selection statistics."""
    selector = build_selector('uniform_random', seed=42)
    selector.initialize(mock_cameras_simple)

    # Make some selections
    n_selections = 50
    for i in range(n_selections):
        cam = selector.select_view(mock_gaussians, iteration=i)

    # Get statistics
    stats = selector.get_selection_statistics()

    assert stats['total_selections'] == n_selections, \
        f"Expected {n_selections} selections, got {stats['total_selections']}"

    assert stats['unique_cameras'] <= len(mock_cameras_simple), \
        "Unique cameras should not exceed total cameras"

    assert stats['most_selected'] is not None, "Should have most selected camera"
    assert stats['least_selected'] is not None, "Should have least selected camera"

    print(f"✓ Selection statistics tracked correctly")
    print(f"  Total: {stats['total_selections']}, Unique: {stats['unique_cameras']}")
    print(f"  Most selected: camera {stats['most_selected'][0]} ({stats['most_selected'][1]} times)")


# Test 9: GaussianAwareSelector with mock Gaussians
@pytest.mark.skip(reason="GaussianAwareSelector not fully implemented/used")
def test_gaussian_aware_selector(mock_cameras_simple, mock_gaussians):
    """Test that GaussianAwareSelector works with mock Gaussians."""
    # Create mock Gaussians with positions
    n_gaussians = 1000
    gaussian_positions = torch.randn(n_gaussians, 3) * 5.0  # Random positions

    # Mock the get_xyz property
    mock_gaussians.get_xyz = gaussian_positions

    # Test inverse_density mode
    config_inverse = {
        'mode': 'inverse_density',
        'update_frequency': 100,
        'temperature': 1.0
    }
    selector = build_selector('gaussian_aware', config=config_inverse, seed=42)
    selector.initialize(mock_cameras_simple)

    # Compute probabilities
    probs = selector.compute_probabilities(mock_gaussians, iteration=0)

    # Check probabilities sum to 1
    prob_sum = sum(probs.values())
    assert abs(prob_sum - 1.0) < 1e-6, f"Probabilities sum to {prob_sum}, not 1.0"

    # Check that camera-Gaussian counts were computed
    assert len(selector.camera_gaussian_counts) == len(mock_cameras_simple)
    assert sum(selector.camera_gaussian_counts.values()) > 0, "No Gaussians visible to any camera"

    print(f"✓ GaussianAwareSelector (inverse_density) working correctly")
    print(f"  Gaussian visibility counts: {list(selector.camera_gaussian_counts.values())[:5]}...")

    # Test coverage_gap mode
    config_coverage = {
        'mode': 'coverage_gap',
        'update_frequency': 100,
        'temperature': 1.0
    }
    selector2 = build_selector('gaussian_aware', config=config_coverage, seed=42)
    selector2.initialize(mock_cameras_simple)

    # Compute probabilities
    probs2 = selector2.compute_probabilities(mock_gaussians, iteration=0)

    # Check probabilities sum to 1
    prob_sum2 = sum(probs2.values())
    assert abs(prob_sum2 - 1.0) < 1e-6, f"Probabilities sum to {prob_sum2}, not 1.0"

    # Check that coverage counts were initialized
    assert selector2.coverage_counts is not None, "Coverage counts not initialized"
    assert selector2.coverage_counts.shape[0] == n_gaussians, "Coverage counts wrong size"

    print(f"✓ GaussianAwareSelector (coverage_gap) working correctly")


# Test 10: GaussianAwareSelector coverage tracking
@pytest.mark.skip(reason="GaussianAwareSelector not fully implemented/used")
def test_gaussian_aware_coverage_tracking(mock_cameras_simple, mock_gaussians):
    """Test that coverage tracking updates correctly."""
    n_gaussians = 100
    gaussian_positions = torch.randn(n_gaussians, 3) * 5.0
    mock_gaussians.get_xyz = gaussian_positions

    config = {
        'mode': 'coverage_gap',
        'update_frequency': 50,
        'temperature': 1.0
    }
    selector = build_selector('gaussian_aware', config=config, seed=42)
    selector.initialize(mock_cameras_simple)

    # Initial probabilities
    probs = selector.compute_probabilities(mock_gaussians, iteration=0)

    # Initial coverage should be all zeros
    assert selector.coverage_counts.sum() == 0, "Initial coverage should be zero"

    # Simulate some selections and update coverage
    for i in range(10):
        cam = selector.select_view(mock_gaussians, iteration=i)
        selector.update_coverage_counts(mock_gaussians, cam)

    # Coverage should have increased
    assert selector.coverage_counts.sum() > 0, "Coverage counts should increase after selections"

    print(f"✓ Coverage tracking working correctly")
    print(f"  Total coverage after 10 selections: {selector.coverage_counts.sum().item():.0f}")


# Test 11: ScheduledHybridSelector with linear schedule
def test_scheduled_hybrid_linear(mock_cameras_simple, mock_gaussians):
    """Test ScheduledHybridSelector with linear weight interpolation."""
    config = {
        'selectors': ['uniform_random', 'geometric'],
        'weights_start': [0.8, 0.2],
        'weights_end': [0.2, 0.8],
        'schedule_type': 'linear',
        'max_iterations': 1000,
    }

    selector = build_selector('scheduled_hybrid', config=config, seed=42)
    selector.initialize(mock_cameras_simple)

    # Test at start (iteration 0)
    probs_start = selector.compute_probabilities(mock_gaussians, iteration=0)
    assert abs(sum(probs_start.values()) - 1.0) < 1e-6, "Probabilities should sum to 1.0"

    # Test at middle (iteration 500)
    probs_mid = selector.compute_probabilities(mock_gaussians, iteration=500)
    assert abs(sum(probs_mid.values()) - 1.0) < 1e-6, "Probabilities should sum to 1.0"

    # Test at end (iteration 1000)
    probs_end = selector.compute_probabilities(mock_gaussians, iteration=1000)
    assert abs(sum(probs_end.values()) - 1.0) < 1e-6, "Probabilities should sum to 1.0"

    # Verify weights changed over time
    weights_start = selector._get_current_weights(0)
    weights_mid = selector._get_current_weights(500)
    weights_end = selector._get_current_weights(1000)

    # At start, should be close to [0.8, 0.2]
    assert abs(weights_start[0] - 0.8) < 0.01
    assert abs(weights_start[1] - 0.2) < 0.01

    # At middle, should be around [0.5, 0.5]
    assert abs(weights_mid[0] - 0.5) < 0.1
    assert abs(weights_mid[1] - 0.5) < 0.1

    # At end, should be close to [0.2, 0.8]
    assert abs(weights_end[0] - 0.2) < 0.01
    assert abs(weights_end[1] - 0.8) < 0.01

    print(f"✓ Linear schedule: {weights_start} → {weights_mid} → {weights_end}")


# Test 12: ScheduledHybridSelector with step schedule
def test_scheduled_hybrid_step(mock_cameras_simple, mock_gaussians):
    """Test ScheduledHybridSelector with step schedule."""
    config = {
        'selectors': ['uniform_random', 'geometric', 'clustering'],
        'schedule_type': 'step',
        'milestones': [
            [0, [0.6, 0.3, 0.1]],
            [500, [0.3, 0.4, 0.3]],
            [1000, [0.1, 0.2, 0.7]],
        ],
    }

    selector = build_selector('scheduled_hybrid', config=config, seed=42)
    selector.initialize(mock_cameras_simple)

    # Test weights at different phases
    weights_phase1 = selector._get_current_weights(0)
    weights_phase2 = selector._get_current_weights(500)
    weights_phase3 = selector._get_current_weights(1000)

    # Phase 1: [0.6, 0.3, 0.1]
    assert abs(weights_phase1[0] - 0.6) < 0.01
    assert abs(weights_phase1[1] - 0.3) < 0.01
    assert abs(weights_phase1[2] - 0.1) < 0.01

    # Phase 2: [0.3, 0.4, 0.3]
    assert abs(weights_phase2[0] - 0.3) < 0.01
    assert abs(weights_phase2[1] - 0.4) < 0.01
    assert abs(weights_phase2[2] - 0.3) < 0.01

    # Phase 3: [0.1, 0.2, 0.7]
    assert abs(weights_phase3[0] - 0.1) < 0.01
    assert abs(weights_phase3[1] - 0.2) < 0.01
    assert abs(weights_phase3[2] - 0.7) < 0.01

    print(f"✓ Step schedule: Phase1={weights_phase1}, Phase2={weights_phase2}, Phase3={weights_phase3}")


# Test 13: ScheduledHybridSelector with standard presets
def test_scheduled_hybrid_presets(mock_cameras_simple, mock_gaussians):
    """Test ScheduledHybridSelector with standard preset configurations.
    
    Note: Some presets use gaussian_aware which requires a real Gaussian model,
    so we skip those presets that would fail with mock data.
    """
    from view_selection import get_standard_config, STANDARD_CONFIGS

    # Presets that use gaussian_aware (requires real Gaussians)
    skip_presets = {'geometric_to_gaussian'}

    # Test available presets that work with mock data
    for preset_name in STANDARD_CONFIGS.keys():
        if preset_name in skip_presets:
            print(f"~ Preset '{preset_name}' skipped (requires real Gaussian model)")
            continue
            
        config = {'preset': preset_name}
        selector = build_selector('scheduled_hybrid', config=config, seed=42)
        selector.initialize(mock_cameras_simple)

        # Compute probabilities at different iterations
        for iteration in [0, 1000, 10000]:
            probs = selector.compute_probabilities(mock_gaussians, iteration=iteration)
            prob_sum = sum(probs.values())
            assert abs(prob_sum - 1.0) < 1e-6, \
                f"Preset '{preset_name}' probabilities sum to {prob_sum}, not 1.0"

        print(f"✓ Preset '{preset_name}' working correctly")


# Test 14: ScheduledHybridSelector forwards update_loss
def test_scheduled_hybrid_forwards_updates(mock_cameras_simple, mock_gaussians):
    """Test that ScheduledHybridSelector forwards updates to sub-selectors."""
    config = {
        'selectors': ['uniform_random', 'loss_based'],
        'weights_start': [0.5, 0.5],
        'weights_end': [0.5, 0.5],
        'schedule_type': 'linear',
        'max_iterations': 1000,
    }

    selector = build_selector('scheduled_hybrid', config=config, seed=42)
    selector.initialize(mock_cameras_simple)

    # Make some selections and update losses
    for i in range(10):
        cam = selector.select_view(mock_gaussians, iteration=i)
        loss = 0.5 + 0.1 * i  # Increasing loss
        selector.update_loss(cam, loss)

    # Find the loss_based sub-selector and check it received updates
    loss_selector = None
    for sub in selector.sub_selectors:
        if hasattr(sub, 'losses'):
            loss_selector = sub
            break

    assert loss_selector is not None, "Should have a loss-based sub-selector"
    assert sum(loss_selector.loss_sample_counts.values()) == 10, \
        "Loss updates should have been forwarded"

    print(f"✓ ScheduledHybridSelector forwards updates correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
