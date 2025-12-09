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
        # Create selector
        selector = build_selector(selector_name, seed=42)

        # Initialize with cameras
        selector.initialize(mock_cameras_simple)

        assert selector.initialized, f"Selector '{selector_name}' failed to initialize"
        print(f"✓ {selector_name} initialized successfully")


# Test 2: compute_probabilities() sums to 1.0
@pytest.mark.parametrize("selector_name", list_selectors())
def test_probability_sum(selector_name, mock_cameras_simple, mock_gaussians):
    """Test that probabilities sum to 1.0 for each selector."""
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
@pytest.mark.parametrize("selector_name", list_selectors())
def test_reproducibility(selector_name, mock_cameras_simple, mock_gaussians):
    """Test that same seed produces same selection sequence."""
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
@pytest.mark.parametrize("selector_name", list_selectors())
def test_all_cameras_nonzero_prob(selector_name, mock_cameras_simple, mock_gaussians):
    """Test that all cameras have non-zero sampling probability."""
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


# Test 5: EpochBasedSelector resets after N selections
def test_epoch_selector_reset(mock_cameras_simple, mock_gaussians):
    """Test that EpochBasedSelector resets selection counts after each epoch."""
    selector = build_selector('epoch_based', seed=42)
    selector.initialize(mock_cameras_simple)

    n_cameras = len(mock_cameras_simple)

    # Track epoch numbers during selections
    epoch_numbers = []

    # Make selections for 2.5 epochs
    n_selections = int(2.5 * n_cameras)
    for i in range(n_selections):
        cam = selector.select_view(mock_gaussians, iteration=i)
        epoch_numbers.append(selector.epoch_number)

    # Check that epoch number increased
    assert max(epoch_numbers) >= 2, "Epoch should have incremented at least twice"

    # Check that epoch resets happened at correct intervals
    # After N selections, we should be in epoch 1
    # After 2N selections, we should be in epoch 2
    assert epoch_numbers[n_cameras] >= 1, f"Should be in epoch 1+ after {n_cameras} selections"
    assert epoch_numbers[2 * n_cameras] >= 2, f"Should be in epoch 2+ after {2*n_cameras} selections"

    print(f"✓ EpochBasedSelector reset correctly. Epochs seen: {set(epoch_numbers)}")


# Test 6: ClusteringSelector assigns all cameras to clusters
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


# Test 7: LossBasedSelector updates loss correctly
def test_loss_based_selector_updates(mock_cameras_simple, mock_gaussians):
    """Test that LossBasedSelector correctly updates EMA losses."""
    config = {
        'ema_decay': 0.9,
        'temperature': 1.0,
        'min_samples_before_bias': 3
    }
    selector = build_selector('loss_based', config=config, seed=42)
    selector.initialize(mock_cameras_simple)

    # Initially all cameras should have zero EMA loss
    assert all(loss == 0.0 for loss in selector.ema_losses.values()), \
        "Initial EMA losses should be zero"

    # Simulate training: select cameras and update losses
    for i in range(30):
        cam = selector.select_view(mock_gaussians, iteration=i)

        # Simulate different loss values
        # Camera 0 has high loss, others have low loss
        loss_value = 1.0 if cam.uid == 0 else 0.1
        selector.update_loss(cam, loss_value)

    # Check that losses have been updated
    assert selector.ema_losses[0] > 0.5, "Camera 0 should have high EMA loss"

    # Check that sample counts increased
    total_samples = sum(selector.loss_sample_counts.values())
    assert total_samples == 30, f"Expected 30 loss updates, got {total_samples}"

    print(f"✓ LossBasedSelector updated losses correctly")
    print(f"  EMA losses: {dict(list(selector.ema_losses.items())[:3])}")


# Test 8: Clustering with DBSCAN handles noise points
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


# Test 9: Selection statistics tracking
def test_selection_statistics(mock_cameras_simple, mock_gaussians):
    """Test that selectors correctly track selection statistics."""
    selector = build_selector('random', seed=42)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
