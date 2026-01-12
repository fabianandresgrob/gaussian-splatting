"""
Debug test for GeometricDiversitySelector.

This test allows step-by-step debugging of the geometric selector with
either mock cameras or real scene data.

Usage:
    # Run with pytest (mock cameras)
    pytest tests/test_geometric_selector_debug.py -v -s

    # Run with real scene (specify path)
    python tests/test_geometric_selector_debug.py --scene /path/to/scene/dslr

    # Run specific test
    pytest tests/test_geometric_selector_debug.py::TestGeometricSelectorDebug::test_static_mode -v -s
"""

import pytest
import numpy as np
import torch
import sys
import os
from typing import List, Dict
from unittest.mock import Mock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from view_selection.geometric_selector import GeometricDiversitySelector


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
# Fixtures
# =============================================================================

@pytest.fixture
def cameras_with_duplicates():
    """
    Create cameras where some are duplicates (same position/orientation).
    
    Layout:
    - 8 unique cameras in a circle
    - 2 duplicate copies of camera 0 (total 3 cameras at same position)
    """
    cameras = []
    n_unique = 8
    radius = 5.0

    for i in range(n_unique):
        angle = 2 * np.pi * i / n_unique
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 1.0

        cos_a, sin_a = np.cos(angle + np.pi), np.sin(angle + np.pi)
        R = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        cam = MockCamera(uid=i, position=np.array([x, y, z]), rotation=R)
        cameras.append(cam)

    # Add duplicates of camera 0
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
def cameras_diverse():
    """Create cameras with varied distances from center (tests distance heuristic)."""
    cameras = []
    
    # Some cameras close to origin
    for i in range(3):
        pos = np.array([i * 0.5, 0, 0])
        cameras.append(MockCamera(uid=i, position=pos))
    
    # Some cameras far from origin
    for i in range(3):
        pos = np.array([10 + i * 0.5, 0, 0])
        cameras.append(MockCamera(uid=3 + i, position=pos))
    
    return cameras


@pytest.fixture
def mock_gaussians():
    """Mock Gaussian model (unused by geometric selector)."""
    return Mock()


# =============================================================================
# Debug Tests
# =============================================================================

class TestGeometricSelectorDebug:
    """Debug tests for GeometricDiversitySelector."""

    def test_static_mode(self, cameras_diverse, mock_gaussians):
        """
        Test static mode: probabilities computed once at init.
        
        Cameras far from scene center should get higher probability.
        """
        cameras = cameras_diverse
        print(f"\n{'='*60}")
        print("GEOMETRIC SELECTOR DEBUG - STATIC MODE")
        print(f"{'='*60}")

        config = {
            "mode": "static",
            "temperature": 1.0,
            "distance_weight": 0.5,
            "diversity_weight": 0.5,
        }
        selector = GeometricDiversitySelector(config=config, verbose=True, seed=42)

        # --- Step 1: Initialize ---
        print(f"\n[Step 1] Initializing with {len(cameras)} cameras...")
        selector.initialize(cameras)

        # --- Step 2: Show camera positions ---
        print(f"\n[Step 2] Camera Positions:")
        positions = selector.positions
        scene_center = np.mean(positions, axis=0)
        print(f"  Scene center: {scene_center}")
        
        for cam in cameras:
            pos = cam.camera_center.numpy()
            dist = np.linalg.norm(pos - scene_center)
            print(f"    Camera {cam.uid}: pos={pos}, dist_to_center={dist:.2f}")

        # --- Step 3: Compute probabilities ---
        print(f"\n[Step 3] Computing Probabilities:")
        probs = selector.compute_probabilities(mock_gaussians, iteration=0)
        
        # Sort by probability
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        for uid, prob in sorted_probs:
            cam = next(c for c in cameras if c.uid == uid)
            pos = cam.camera_center.numpy()
            dist = np.linalg.norm(pos - scene_center)
            print(f"    Camera {uid}: prob={prob:.4f}, dist={dist:.2f}")

        # Verify probabilities sum to 1
        prob_sum = sum(probs.values())
        print(f"\n  Probability sum: {prob_sum:.6f}")
        assert abs(prob_sum - 1.0) < 1e-6, f"Probabilities should sum to 1, got {prob_sum}"
        print("  ✓ Probabilities sum to 1.0")

        # --- Step 4: Verify static behavior ---
        print(f"\n[Step 4] Verifying static mode (probs unchanged after selection):")
        selected = selector.select_view(mock_gaussians, iteration=0)
        print(f"  Selected: Camera {selected.uid}")
        
        probs_after = selector.compute_probabilities(mock_gaussians, iteration=1)
        
        changed = False
        for uid in probs:
            if abs(probs[uid] - probs_after[uid]) > 1e-6:
                changed = True
                break
        
        if not changed:
            print("  ✓ Probabilities unchanged (static mode working)")
        else:
            print("  ✗ Probabilities changed unexpectedly!")

        print(f"\n{'='*60}")
        print("DEBUG COMPLETE")
        print(f"{'='*60}")

    def test_dynamic_mode(self, cameras_with_duplicates, mock_gaussians):
        """
        Test dynamic mode (distance_to_selected): probabilities adapt based on recent selections.
        
        Key check: After selecting a camera, nearby cameras should have LOWER probability.
        """
        cameras = cameras_with_duplicates
        print(f"\n{'='*60}")
        print("GEOMETRIC SELECTOR DEBUG - DYNAMIC MODE")
        print(f"{'='*60}")

        config = {
            "mode": "distance_to_selected",
            "temperature": 0.5,
            "recency_window": 5,
            "use_cumulative_penalty": False,
        }
        selector = GeometricDiversitySelector(config=config, verbose=True, seed=42)

        # --- Step 1: Initialize ---
        print(f"\n[Step 1] Initializing with {len(cameras)} cameras...")
        selector.initialize(cameras)

        # --- Step 2: Initial probabilities (before any selection) ---
        print(f"\n[Step 2] Initial Probabilities (no prior selections):")
        probs_initial = selector.compute_probabilities(mock_gaussians, iteration=0)
        
        for cam in cameras:
            pos = cam.camera_center.numpy()
            print(f"    Camera {cam.uid:2d}: prob={probs_initial[cam.uid]:.4f}, pos={pos}")

        # Verify sum
        prob_sum = sum(probs_initial.values())
        print(f"\n  Probability sum: {prob_sum:.6f}")
        assert abs(prob_sum - 1.0) < 1e-6
        print("  ✓ Probabilities sum to 1.0")

        # --- Step 3: Select camera 0 and check how probabilities change ---
        print(f"\n[Step 3] Selecting Camera 0 and checking probability changes:")
        
        # Manually select camera 0 to track its effect
        cam_0 = cameras[0]
        selector.recent_selections.append(0)  # Add to recent selections
        
        probs_after = selector.compute_probabilities(mock_gaussians, iteration=1)
        
        print(f"\n  After selecting Camera 0:")
        print(f"  Camera 0 position: {cam_0.camera_center.numpy()}")
        
        # Find duplicates (same position as camera 0)
        duplicate_uids = [cam.uid for cam in cameras if "__dup" in cam.image_name]
        
        for cam in cameras:
            before = probs_initial[cam.uid]
            after = probs_after[cam.uid]
            change = after - before
            marker = "↓" if change < -0.01 else ("↑" if change > 0.01 else "=")
            is_dup = cam.uid in duplicate_uids or cam.uid == 0
            dup_marker = " [DUP/ORIG]" if is_dup else ""
            print(f"    Camera {cam.uid:2d}: {before:.4f} -> {after:.4f} ({marker}){dup_marker}")

        # --- Step 4: Check duplicates have lower probability ---
        print(f"\n[Step 4] Checking duplicate handling:")
        
        # Duplicates should have lower probability since they're at same position
        orig_prob = probs_after[0]
        dup_probs = [probs_after[uid] for uid in duplicate_uids]
        
        print(f"  Camera 0 (original, selected): prob={orig_prob:.4f}")
        for uid in duplicate_uids:
            print(f"  Camera {uid} (duplicate): prob={probs_after[uid]:.4f}")
        
        # Since duplicates are at exact same position as camera 0, they should
        # have the same (reduced) probability after camera 0 is selected
        if dup_probs and all(abs(p - orig_prob) < 0.01 for p in dup_probs):
            print("  ✓ Duplicates have same probability as original (correct - same position)")
        else:
            print("  ≈ Duplicates have different probability (check distance calculation)")

        # --- Step 5: Run many iterations and track selection distribution ---
        print(f"\n[Step 5] Running 500 iterations to check selection distribution:")
        
        # Reset and run fresh
        selector2 = GeometricDiversitySelector(config=config, verbose=False, seed=42)
        selector2.initialize(cameras)
        
        selection_counts = {cam.uid: 0 for cam in cameras}
        n_iter = 500
        
        for i in range(n_iter):
            cam = selector2.select_view(mock_gaussians, iteration=i)
            selection_counts[cam.uid] += 1
        
        print(f"\n  Selection counts:")
        total_dup_pos = 0
        for cam in cameras:
            count = selection_counts[cam.uid]
            pct = 100 * count / n_iter
            is_dup = "__dup" in cam.image_name or cam.uid == 0
            marker = " [DUP/ORIG]" if is_dup else ""
            print(f"    Camera {cam.uid:2d}: {count:4d} ({pct:5.1f}%){marker}")
            if is_dup:
                total_dup_pos += count
        
        n_dup_cams = len(duplicate_uids) + 1
        expected_uniform = 100 * n_dup_cams / len(cameras)
        actual = 100 * total_dup_pos / n_iter
        
        print(f"\n  Summary:")
        print(f"    Selections at duplicated position: {total_dup_pos}/{n_iter} = {actual:.1f}%")
        print(f"    Expected if uniform: {expected_uniform:.1f}%")
        
        if actual < expected_uniform * 0.8:
            print("    ✓ Geometric selector is AVOIDING duplicated position!")
        elif actual > expected_uniform * 1.2:
            print("    ✗ Geometric selector is OVER-selecting duplicated position!")
        else:
            print("    ≈ Similar to uniform random")

        print(f"\n{'='*60}")
        print("DEBUG COMPLETE")
        print(f"{'='*60}")

    def test_recency_window_effect(self, cameras_with_duplicates, mock_gaussians):
        """Test that recency_window controls how long selections are remembered."""
        cameras = cameras_with_duplicates
        print(f"\n{'='*60}")
        print("GEOMETRIC SELECTOR - RECENCY WINDOW TEST")
        print(f"{'='*60}")

        # Small recency window
        config = {
            "mode": "distance_to_selected",
            "temperature": 1.0,
            "recency_window": 3,  # Only remember last 3 selections
            "use_cumulative_penalty": False,
        }
        selector = GeometricDiversitySelector(config=config, verbose=True, seed=42)
        selector.initialize(cameras)

        print(f"\n  Recency window: {selector.recency_window}")
        
        # Select camera 0 multiple times, then select others
        print(f"\n  Selecting cameras: 0, 0, 0, 1, 2, 3, 4")
        
        for i, uid in enumerate([0, 0, 0, 1, 2, 3, 4]):
            cam = cameras[uid]
            selector.recent_selections.append(uid)
            if len(selector.recent_selections) > selector.recency_window:
                selector.recent_selections = selector.recent_selections[-selector.recency_window:]
            
            print(f"    After selecting {uid}: recent_selections = {selector.recent_selections}")
        
        # After selecting 1, 2, 3, 4, the old selections of camera 0 should be forgotten
        # So camera 0's probability should recover
        probs = selector.compute_probabilities(mock_gaussians, iteration=7)
        
        print(f"\n  Final probabilities (camera 0's selections should be 'forgotten'):")
        for cam in cameras[:5]:
            print(f"    Camera {cam.uid}: prob={probs[cam.uid]:.4f}")

        print(f"\n{'='*60}")


# =============================================================================
# Real Scene Loading
# =============================================================================

def load_cameras_from_scene(scene_path: str) -> List:
    """Load real cameras from a scene."""
    from argparse import Namespace
    from scene.dataset_readers import sceneLoadTypeCallbacks
    from utils.camera_utils import cameraList_from_camInfos
    
    is_colmap = os.path.exists(os.path.join(scene_path, "sparse"))
    is_scannetpp = (
        os.path.exists(os.path.join(scene_path, "nerfstudio", "transforms_undistorted.json"))
        or os.path.exists(os.path.join(scene_path, "nerfstudio", "transforms.json"))
    )
    is_blender = os.path.exists(os.path.join(scene_path, "transforms_train.json"))
    
    if is_colmap:
        print("Detected COLMAP scene")
        scene_info = sceneLoadTypeCallbacks["Colmap"](scene_path, "images", "", False, False)
        is_nerf_synthetic = False
    elif is_scannetpp:
        print("Detected ScanNet++ scene")
        from dataset import readScannetppInfo
        scene_info = readScannetppInfo(scene_path)
        is_nerf_synthetic = False
    elif is_blender:
        print("Detected Blender scene")
        scene_info = sceneLoadTypeCallbacks["Blender"](scene_path, False, "", False)
        is_nerf_synthetic = True
    else:
        raise ValueError(f"Could not detect scene type at {scene_path}")
    
    args = Namespace(resolution=8, data_device="cpu", train_test_exp=False)
    cameras = cameraList_from_camInfos(
        scene_info.train_cameras, resolution_scale=1.0, args=args,
        is_nerf_synthetic=is_nerf_synthetic, is_test_dataset=False
    )
    return cameras


def run_debug_with_real_scene(scene_path: str):
    """Run geometric selector debug with a real scene."""
    from collections import defaultdict
    
    print(f"\n{'='*60}")
    print(f"GEOMETRIC SELECTOR - REAL SCENE: {scene_path}")
    print(f"{'='*60}")

    try:
        cameras = load_cameras_from_scene(scene_path)
        print(f"Loaded {len(cameras)} cameras")
    except Exception as e:
        print(f"Failed to load scene: {e}")
        return

    config = {
        "mode": "distance_to_selected",
        "temperature": 0.3,
        "recency_window": 500,
        "use_cumulative_penalty": False,
    }
    selector = GeometricDiversitySelector(config=config, verbose=True, seed=42)
    selector.initialize(cameras)

    mock_gaussians = Mock()

    # Initial probabilities
    print(f"\nInitial probability distribution:")
    probs = selector.compute_probabilities(mock_gaussians, iteration=0)
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    print(f"  Top 5 cameras:")
    for uid, prob in sorted_probs[:5]:
        cam = next(c for c in cameras if c.uid == uid)
        print(f"    Camera {uid} ({cam.image_name}): prob={prob:.4f}")
    
    print(f"  Bottom 5 cameras:")
    for uid, prob in sorted_probs[-5:]:
        cam = next(c for c in cameras if c.uid == uid)
        print(f"    Camera {uid} ({cam.image_name}): prob={prob:.4f}")

    # Check for duplicates
    print(f"\n--- Duplicate Camera Check ---")
    duplicate_cameras = [cam for cam in cameras if "__dup" in cam.image_name]
    
    if duplicate_cameras:
        dup_groups = defaultdict(list)
        for cam in cameras:
            base_name = cam.image_name.split("__dup")[0]
            dup_groups[base_name].append(cam)
        
        for base_name, group_cams in dup_groups.items():
            if len(group_cams) > 1:
                group_probs = [probs[c.uid] for c in group_cams]
                print(f"  '{base_name}': {len(group_cams)} cameras, probs={[f'{p:.4f}' for p in group_probs]}")
    else:
        print(f"  No duplicate cameras found")

    # Run iterations
    print(f"\nRunning 500 iterations...")
    selection_counts = {cam.uid: 0 for cam in cameras}
    n_iter = 500
    
    for i in range(n_iter):
        cam = selector.select_view(mock_gaussians, iteration=i)
        selection_counts[cam.uid] += 1

    # Analyze
    print(f"\nSelection distribution (top 10):")
    sorted_counts = sorted(selection_counts.items(), key=lambda x: x[1], reverse=True)
    for uid, count in sorted_counts[:10]:
        cam = next(c for c in cameras if c.uid == uid)
        pct = 100 * count / n_iter
        marker = " [DUP]" if "__dup" in cam.image_name else ""
        print(f"  Camera {uid} ({cam.image_name}): {count} ({pct:.1f}%){marker}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug GeometricDiversitySelector")
    parser.add_argument("--scene", type=str, help="Path to a real scene directory")
    parser.add_argument("--pytest", action="store_true", help="Run pytest instead")
    args = parser.parse_args()

    if args.pytest:
        pytest.main([__file__, "-v", "-s"])
    elif args.scene:
        run_debug_with_real_scene(args.scene)
    else:
        print("Running with mock cameras (use --scene /path/to/scene for real data)")
        pytest.main([__file__, "-v", "-s"])
