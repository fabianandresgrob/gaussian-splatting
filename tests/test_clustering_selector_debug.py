"""
Debug test for ClusteringSelector.

This test allows step-by-step debugging of the clustering selector with
either mock cameras or real scene data.

Usage:
    # Run with pytest (mock cameras)
    pytest tests/test_clustering_selector_debug.py -v -s

    # Run with real scene (specify path)
    python tests/test_clustering_selector_debug.py --scene /path/to/scene/dslr

    # Run specific test
    pytest tests/test_clustering_selector_debug.py::test_clustering_single_iteration -v -s
"""

import pytest
import numpy as np
import torch
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
from unittest.mock import Mock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from view_selection.clustering_selector import ClusteringSelector


# =============================================================================
# Mock Camera Class
# =============================================================================

class MockCamera:
    """Mock camera object matching the real Camera interface."""

    def __init__(
        self,
        uid: int,
        position: np.ndarray,
        rotation: np.ndarray = None,
        image_name: str = None
    ):
        self.uid = uid
        self.image_name = image_name or f"image_{uid:04d}.jpg"
        self.camera_center = torch.tensor(position, dtype=torch.float32)

        if rotation is None:
            self.R = np.eye(3, dtype=np.float32)
        else:
            self.R = rotation.astype(np.float32)


# =============================================================================
# Fixtures: Various Camera Configurations
# =============================================================================

@pytest.fixture
def cameras_with_duplicates():
    """
    Create cameras where some are duplicates (same position/orientation).
    
    Layout:
    - 8 unique cameras in a circle
    - 2 duplicate copies of camera 0 (total 3 cameras at same position)
    
    This simulates the imbalanced dataset scenario.
    """
    cameras = []
    n_unique = 8
    radius = 5.0

    # Create unique cameras in a circle
    for i in range(n_unique):
        angle = 2 * np.pi * i / n_unique
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 1.0

        # Rotation looking toward center
        cos_a, sin_a = np.cos(angle + np.pi), np.sin(angle + np.pi)
        R = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        cam = MockCamera(uid=i, position=np.array([x, y, z]), rotation=R)
        cameras.append(cam)

    # Add duplicates of camera 0 (same position, same rotation)
    original_pos = cameras[0].camera_center.numpy()
    original_R = cameras[0].R

    for dup_idx in range(2):
        dup_cam = MockCamera(
            uid=n_unique + dup_idx,
            position=original_pos.copy(),
            rotation=original_R.copy(),
            image_name=f"image_0000__dup{dup_idx + 1}.jpg"
        )
        cameras.append(dup_cam)

    return cameras


@pytest.fixture
def cameras_clustered():
    """
    Create cameras that naturally form 3 clusters.
    
    Cluster 0: 5 cameras around (0, 0, 0)
    Cluster 1: 3 cameras around (10, 0, 0)
    Cluster 2: 2 cameras around (0, 10, 0)
    
    This tests that clustering correctly identifies groups.
    """
    cameras = []
    uid = 0

    # Cluster 0: 5 cameras near origin
    for i in range(5):
        offset = np.random.RandomState(i).randn(3) * 0.5
        cam = MockCamera(uid=uid, position=np.array([0, 0, 0]) + offset)
        cameras.append(cam)
        uid += 1

    # Cluster 1: 3 cameras near (10, 0, 0)
    for i in range(3):
        offset = np.random.RandomState(100 + i).randn(3) * 0.5
        cam = MockCamera(uid=uid, position=np.array([10, 0, 0]) + offset)
        cameras.append(cam)
        uid += 1

    # Cluster 2: 2 cameras near (0, 10, 0)
    for i in range(2):
        offset = np.random.RandomState(200 + i).randn(3) * 0.5
        cam = MockCamera(uid=uid, position=np.array([0, 10, 0]) + offset)
        cameras.append(cam)
        uid += 1

    return cameras


@pytest.fixture
def mock_gaussians():
    """Mock Gaussian model (unused by clustering selector)."""
    return Mock()


# =============================================================================
# Debug Tests
# =============================================================================

class TestClusteringSelectorDebug:
    """Debug tests for step-by-step inspection of ClusteringSelector."""

    def test_clustering_single_iteration(self, cameras_with_duplicates, mock_gaussians):
        """
        Debug a single iteration of the clustering selector.
        
        Step through this test to inspect:
        1. How cameras are clustered
        2. Initial probability distribution
        3. Selection and probability update
        """
        cameras = cameras_with_duplicates
        print(f"\n{'='*60}")
        print("CLUSTERING SELECTOR DEBUG - SINGLE ITERATION")
        print(f"{'='*60}")

        # --- Step 1: Create selector with KMeans ---
        config = {
            "clustering_method": "kmeans",
            "n_clusters": 4,
            "temperature": 1.0,
            "use_orientation": True,
        }
        selector = ClusteringSelector(config=config, verbose=True, seed=42)

        # --- Step 2: Initialize (clustering happens here) ---
        print(f"\n[Step 1] Initializing with {len(cameras)} cameras...")
        selector.initialize(cameras)

        # Print cluster assignments
        print(f"\n[Step 2] Cluster Assignments:")
        print(f"  Number of clusters: {selector.n_clusters}")
        print(f"  Cluster sizes: {selector.cluster_sizes}")

        for cam in cameras:
            cluster_id = selector.camera_clusters[cam.uid]
            pos = cam.camera_center.numpy()
            print(f"    Camera {cam.uid:2d} ({cam.image_name:25s}) -> Cluster {cluster_id} | pos={pos}")

        # Check: duplicates should be in same cluster
        dup_clusters = [selector.camera_clusters[cam.uid] for cam in cameras if "__dup" in cam.image_name]
        orig_cluster = selector.camera_clusters[0]  # Original camera 0
        print(f"\n  Original camera 0 cluster: {orig_cluster}")
        print(f"  Duplicate cameras clusters: {dup_clusters}")
        if dup_clusters:
            assert all(c == orig_cluster for c in dup_clusters), \
                "FAIL: Duplicates should be in same cluster as original!"
            print("  ✓ All duplicates correctly in same cluster")

        # --- Step 3: Compute initial probabilities ---
        print(f"\n[Step 3] Initial Probabilities:")
        probs = selector.compute_probabilities(mock_gaussians, iteration=0)

        # Sort by cluster for readability
        by_cluster: Dict[int, List] = {}
        for cam in cameras:
            c = selector.camera_clusters[cam.uid]
            by_cluster.setdefault(c, []).append((cam.uid, probs[cam.uid], cam.image_name))

        for cluster_id in sorted(by_cluster.keys()):
            cams_in_cluster = by_cluster[cluster_id]
            total_prob = sum(p for _, p, _ in cams_in_cluster)
            print(f"\n  Cluster {cluster_id} (size={selector.cluster_sizes[cluster_id]}, total_prob={total_prob:.4f}):")
            for uid, prob, name in cams_in_cluster:
                print(f"    Camera {uid:2d}: prob={prob:.4f} ({name})")

        # Verify probabilities sum to 1
        prob_sum = sum(probs.values())
        print(f"\n  Probability sum: {prob_sum:.6f}")
        assert abs(prob_sum - 1.0) < 1e-6, f"Probabilities should sum to 1, got {prob_sum}"
        print("  ✓ Probabilities sum to 1.0")

        # --- Step 4: Make a selection ---
        print(f"\n[Step 4] Making first selection...")
        selected_cam = selector.select_view(mock_gaussians, iteration=0)
        selected_cluster = selector.camera_clusters[selected_cam.uid]
        print(f"  Selected: Camera {selected_cam.uid} ({selected_cam.image_name})")
        print(f"  From cluster: {selected_cluster}")

        # --- Step 5: Verify probabilities are static (no dynamic updates) ---
        print(f"\n[Step 5] Verifying probabilities are static (iteration 1):")
        probs_after = selector.compute_probabilities(mock_gaussians, iteration=1)

        # Probabilities should be unchanged (static mode)
        print(f"\n  Probability comparison (should be unchanged - static mode):")
        probs_changed = False
        for cluster_id in sorted(by_cluster.keys()):
            cams_in_cluster = by_cluster[cluster_id]
            before_total = sum(probs[uid] for uid, _, _ in cams_in_cluster)
            after_total = sum(probs_after[uid] for uid, _, _ in cams_in_cluster)
            change = after_total - before_total
            marker = "↓" if change < -0.001 else ("↑" if change > 0.001 else "=")
            if abs(change) > 0.001:
                probs_changed = True
            print(f"    Cluster {cluster_id}: {before_total:.4f} -> {after_total:.4f} ({marker})")
        
        if not probs_changed:
            print("  ✓ Probabilities are static (unchanged after selection)")

        print(f"\n{'='*60}")
        print("DEBUG COMPLETE")
        print(f"{'='*60}")

    def test_clustering_many_iterations(self, cameras_with_duplicates, mock_gaussians):
        """
        Run many iterations and track selection distribution.
        
        Key question: Does the selector avoid selecting duplicates?
        """
        cameras = cameras_with_duplicates
        n_iterations = 1000

        print(f"\n{'='*60}")
        print(f"CLUSTERING SELECTOR - {n_iterations} ITERATIONS")
        print(f"{'='*60}")

        config = {
            "clustering_method": "kmeans",
            "n_clusters": 4,
            "temperature": 1.0,
            "use_orientation": True,
        }
        selector = ClusteringSelector(config=config, verbose=False, seed=42)
        selector.initialize(cameras)

        # Track selections
        selection_counts = {cam.uid: 0 for cam in cameras}

        for i in range(n_iterations):
            cam = selector.select_view(mock_gaussians, iteration=i)
            selection_counts[cam.uid] += 1

        # Analyze results
        print(f"\nSelection counts after {n_iterations} iterations:")

        dup_selections = 0
        total_selections = n_iterations
        duplicate_uids = [cam.uid for cam in cameras if "__dup" in cam.image_name]
        original_uid = 0  # The duplicated camera

        for cam in cameras:
            count = selection_counts[cam.uid]
            pct = 100 * count / n_iterations
            is_dup = "__dup" in cam.image_name
            is_orig = cam.uid == original_uid
            marker = " [DUP]" if is_dup else (" [ORIG]" if is_orig else "")
            print(f"  Camera {cam.uid:2d}: {count:4d} ({pct:5.1f}%){marker}")

            if is_dup:
                dup_selections += count

        # Include original in "duplicate position" count
        orig_selections = selection_counts[original_uid]
        total_at_dup_position = dup_selections + orig_selections
        n_cameras_at_dup_position = len(duplicate_uids) + 1  # duplicates + original

        print(f"\n--- Summary ---")
        print(f"Cameras at duplicated position: {n_cameras_at_dup_position} (uids: {[original_uid] + duplicate_uids})")
        print(f"Selections at duplicated position: {total_at_dup_position} / {n_iterations} = {100*total_at_dup_position/n_iterations:.1f}%")

        # Expected if uniform random: 3/10 = 30%
        expected_uniform = 100 * n_cameras_at_dup_position / len(cameras)
        print(f"Expected if uniform random: {expected_uniform:.1f}%")

        # With clustering, duplicates should be in same cluster, so cluster gets
        # same probability regardless of size -> individual cameras split it
        # Key insight: cluster probability / cluster size means each dup gets LESS
        print(f"\n--- Analysis ---")
        if total_at_dup_position / n_iterations < expected_uniform / 100 * 0.8:
            print("✓ Clustering is AVOIDING the duplicated position (good!)")
        elif total_at_dup_position / n_iterations > expected_uniform / 100 * 1.2:
            print("✗ Clustering is OVER-selecting the duplicated position (bad!)")
        else:
            print("≈ Clustering selection is similar to uniform random")

    def test_dbscan_clustering(self, cameras_clustered, mock_gaussians):
        """Test DBSCAN clustering which adapts cluster count automatically."""
        cameras = cameras_clustered

        print(f"\n{'='*60}")
        print("DBSCAN CLUSTERING DEBUG")
        print(f"{'='*60}")

        config = {
            "clustering_method": "dbscan",
            "eps": 2.0,  # Neighborhood radius
            "min_samples": 2,
            "temperature": 1.0,
            "use_orientation": False,  # Just position for this test
        }
        selector = ClusteringSelector(config=config, verbose=True, seed=42)
        selector.initialize(cameras)

        print(f"\nDBSCAN found {selector.n_clusters} clusters")
        print(f"Cluster sizes: {selector.cluster_sizes}")

        # Print assignments
        print(f"\nCluster assignments:")
        for cam in cameras:
            cluster_id = selector.camera_clusters[cam.uid]
            pos = cam.camera_center.numpy()
            print(f"  Camera {cam.uid:2d} -> Cluster {cluster_id} | pos={pos}")

        # Compute probabilities
        probs = selector.compute_probabilities(mock_gaussians, iteration=0)

        print(f"\nProbabilities by cluster:")
        by_cluster: Dict[int, List] = {}
        for cam in cameras:
            c = selector.camera_clusters[cam.uid]
            by_cluster.setdefault(c, []).append((cam.uid, probs[cam.uid]))

        for cluster_id in sorted(by_cluster.keys()):
            cams = by_cluster[cluster_id]
            total = sum(p for _, p in cams)
            print(f"  Cluster {cluster_id} (size={len(cams)}): total_prob={total:.4f}")
            for uid, p in cams:
                print(f"    Camera {uid}: {p:.4f}")

    def test_dbscan_noise_points_as_singletons(self, mock_gaussians):
        """
        Test that DBSCAN noise points are treated as individual singleton clusters.
        
        This is critical for imbalanced datasets where unique viewpoints (noise points)
        should receive high sampling probability, not be lumped into one cluster.
        """
        print(f"\n{'='*60}")
        print("DBSCAN NOISE POINTS AS SINGLETON CLUSTERS")
        print(f"{'='*60}")

        # Create cameras: one tight cluster + several isolated "noise" points
        cameras = []
        
        # Cluster 0: 5 cameras very close together (will form a cluster)
        for i in range(5):
            offset = np.random.RandomState(i).randn(3) * 0.1  # Very tight
            cam = MockCamera(uid=i, position=np.array([0, 0, 0]) + offset)
            cameras.append(cam)
        
        # Add 3 isolated noise points (far apart, won't form clusters)
        noise_positions = [
            np.array([100, 0, 0]),   # Far from cluster 0
            np.array([0, 100, 0]),   # Far from everything
            np.array([50, 50, 50]),  # Also isolated
        ]
        for i, pos in enumerate(noise_positions):
            cam = MockCamera(uid=5 + i, position=pos)
            cameras.append(cam)
        
        # Use tight DBSCAN parameters to ensure noise points
        config = {
            "clustering_method": "dbscan",
            "eps": 0.5,       # Small radius
            "min_samples": 3,  # Requires 3 neighbors
            "temperature": 1.0,
            "use_orientation": False,
        }
        selector = ClusteringSelector(config=config, verbose=True, seed=42)
        selector.initialize(cameras)
        
        print(f"\nTotal clusters: {selector.n_clusters}")
        print(f"Cluster sizes: {selector.cluster_sizes}")
        
        # Get statistics
        stats = selector.get_cluster_statistics()
        n_singletons = stats['n_singleton_clusters']
        n_regular = stats['n_regular_clusters']
        
        print(f"Regular clusters: {n_regular}")
        print(f"Singleton clusters (noise points): {n_singletons}")
        
        # Verify: the 3 noise points should each be in their own singleton cluster
        assert n_singletons == 3, f"Expected 3 singleton clusters for noise points, got {n_singletons}"
        
        # Compute probabilities
        probs = selector.compute_probabilities(mock_gaussians, iteration=0)
        
        print("\nProbabilities:")
        cluster_probs = {}
        for cam in cameras:
            c = selector.camera_clusters[cam.uid]
            cluster_probs.setdefault(c, []).append((cam.uid, probs[cam.uid]))
        
        for cluster_id in sorted(cluster_probs.keys()):
            cams_in_cluster = cluster_probs[cluster_id]
            total = sum(p for _, p in cams_in_cluster)
            size = len(cams_in_cluster)
            print(f"  Cluster {cluster_id} (size={size}): total_prob={total:.4f}")
            for uid, p in cams_in_cluster:
                print(f"    Camera {uid}: prob={p:.4f}")
        
        # Key test: each noise point (singleton) should have HIGHER probability than
        # each camera in the large cluster
        large_cluster_cam_prob = probs[0]  # Camera 0 is in the large cluster
        noise_point_prob = probs[5]  # Camera 5 is a noise point (singleton)
        
        print(f"\nLarge cluster camera prob: {large_cluster_cam_prob:.4f}")
        print(f"Noise point (singleton) prob: {noise_point_prob:.4f}")
        
        assert noise_point_prob > large_cluster_cam_prob, \
            f"Noise points should have higher probability than cameras in large clusters! " \
            f"Got noise={noise_point_prob:.4f} vs cluster_cam={large_cluster_cam_prob:.4f}"
        
        print("\n✓ Noise points correctly receive higher sampling probability")


# =============================================================================
# Real Scene Loading (Optional)
# =============================================================================

def load_cameras_from_scene(scene_path: str) -> List:
    """
    Load real cameras from a ScanNet++ or COLMAP scene.
    
    Args:
        scene_path: Path to scene directory (e.g., /path/to/scene/dslr)
        
    Returns:
        List of Camera objects
    """
    from argparse import Namespace
    from scene.dataset_readers import sceneLoadTypeCallbacks
    from utils.camera_utils import cameraList_from_camInfos
    
    # Detect scene type and load scene_info
    is_colmap = os.path.exists(os.path.join(scene_path, "sparse"))
    is_scannetpp = (
        os.path.exists(os.path.join(scene_path, "nerfstudio", "transforms_undistorted.json"))
        or os.path.exists(os.path.join(scene_path, "nerfstudio", "transforms.json"))
    )
    is_blender = os.path.exists(os.path.join(scene_path, "transforms_train.json"))
    
    if is_colmap:
        print("Detected COLMAP scene")
        scene_info = sceneLoadTypeCallbacks["Colmap"](
            scene_path,
            "images",  # images folder
            "",        # depths
            False,     # eval
            False,     # train_test_exp
        )
        is_nerf_synthetic = False
    elif is_scannetpp:
        print("Detected ScanNet++ scene")
        from dataset import readScannetppInfo
        scene_info = readScannetppInfo(scene_path)
        is_nerf_synthetic = False
    elif is_blender:
        print("Detected Blender scene")
        scene_info = sceneLoadTypeCallbacks["Blender"](
            scene_path,
            False,  # white_background
            "",     # depths
            False,  # eval
        )
        is_nerf_synthetic = True
    else:
        raise ValueError(f"Could not detect scene type at {scene_path}")
    
    # Create minimal args object for camera loading
    args = Namespace(
        resolution=8,  # 1/8 resolution for fast loading
        data_device="cpu",
        train_test_exp=False,
    )
    
    # Convert CameraInfo to Camera objects
    cameras = cameraList_from_camInfos(
        scene_info.train_cameras,
        resolution_scale=1.0,
        args=args,
        is_nerf_synthetic=is_nerf_synthetic,
        is_test_dataset=False,
    )
    
    return cameras


def run_debug_with_real_scene(scene_path: str):
    """
    Run clustering debug with a real scene.
    
    Usage:
        python test_clustering_selector_debug.py --scene /path/to/scene
    """
    print(f"\n{'='*60}")
    print(f"LOADING REAL SCENE: {scene_path}")
    print(f"{'='*60}")

    try:
        cameras = load_cameras_from_scene(scene_path)
        print(f"Loaded {len(cameras)} cameras")
    except Exception as e:
        print(f"Failed to load scene: {e}")
        print("Falling back to mock cameras...")
        # Create mock cameras as fallback
        cameras = []
        for i in range(20):
            angle = 2 * np.pi * i / 20
            pos = np.array([5 * np.cos(angle), 5 * np.sin(angle), 1.0])
            cameras.append(MockCamera(uid=i, position=pos))

    # Run clustering
    config = {
        "clustering_method": "kmeans",
        "eps": 0.85,
        "n_clusters": 6,
        "min_samples": 3,
        "temperature": 1.0,
        "use_orientation": True,
    }
    selector = ClusteringSelector(config=config, verbose=True, seed=42)
    selector.initialize(cameras)

    print(f"\nClustering results:")
    print(f"  Method: {selector.clustering_method}")
    print(f"  Number of clusters: {selector.n_clusters}")
    print(f"  Cluster sizes: {selector.cluster_sizes}")

    # Show a few sample cameras per cluster
    by_cluster: Dict[int, List] = {}
    for cam in cameras:
        c = selector.camera_clusters[cam.uid]
        by_cluster.setdefault(c, []).append(cam)

    for cluster_id in sorted(by_cluster.keys()):
        cams = by_cluster[cluster_id]
        print(f"\n  Cluster {cluster_id} ({len(cams)} cameras):")
        for cam in cams[:3]:  # Show first 3
            pos = cam.camera_center.cpu().numpy() if hasattr(cam.camera_center, 'cpu') else cam.camera_center.numpy()
            print(f"    {cam.image_name}: pos={pos}")
        if len(cams) > 3:
            print(f"    ... and {len(cams) - 3} more")

    # Check if duplicated cameras are in the same cluster
    print(f"\n--- Duplicate Camera Check ---")
    duplicate_cameras = [cam for cam in cameras if "__dup" in cam.image_name]
    
    if duplicate_cameras:
        # Group duplicates by their original (strip __dupN suffix to get base name)
        from collections import defaultdict
        dup_groups = defaultdict(list)
        
        for cam in cameras:
            # Extract base name (without __dupN suffix)
            base_name = cam.image_name.split("__dup")[0]
            dup_groups[base_name].append(cam)
        
        # Check groups with more than one camera (i.e., have duplicates)
        all_dups_same_cluster = True
        for base_name, group_cams in dup_groups.items():
            if len(group_cams) > 1:
                clusters = [selector.camera_clusters[c.uid] for c in group_cams]
                unique_clusters = set(clusters)
                
                if len(unique_clusters) == 1:
                    print(f"  ✓ '{base_name}' and its {len(group_cams)-1} duplicate(s) -> all in cluster {clusters[0]}")
                else:
                    print(f"  ✗ '{base_name}' duplicates are SPLIT across clusters: {clusters}")
                    all_dups_same_cluster = False
        
        if all_dups_same_cluster:
            print(f"\n  ✓ All duplicated cameras are correctly in the same cluster as their originals!")
        else:
            print(f"\n  ✗ WARNING: Some duplicates are in different clusters than their originals!")
    else:
        print(f"  No duplicate cameras found (no '__dup' in image names)")

    # Run a few iterations
    mock_gaussians = Mock()
    n_iter = 100
    selection_counts = {cam.uid: 0 for cam in cameras}

    for i in range(n_iter):
        cam = selector.select_view(mock_gaussians, iteration=i)
        selection_counts[cam.uid] += 1

    print(f"\n\nSelection distribution after {n_iter} iterations:")
    for cluster_id in sorted(by_cluster.keys()):
        cams = by_cluster[cluster_id]
        cluster_selections = sum(selection_counts[c.uid] for c in cams)
        pct = 100 * cluster_selections / n_iter
        print(f"  Cluster {cluster_id}: {cluster_selections} selections ({pct:.1f}%)")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug ClusteringSelector")
    parser.add_argument("--scene", type=str, help="Path to a real scene directory")
    parser.add_argument("--pytest", action="store_true", help="Run pytest instead")
    args = parser.parse_args()

    if args.pytest:
        pytest.main([__file__, "-v", "-s"])
    elif args.scene:
        run_debug_with_real_scene(args.scene)
    else:
        # Run with mock cameras
        print("Running with mock cameras (use --scene /path/to/scene for real data)")
        print("Running pytest...")
        pytest.main([__file__, "-v", "-s"])
