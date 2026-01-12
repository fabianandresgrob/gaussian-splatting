"""
Debug test for DINOSelector.

This test allows step-by-step debugging of the DINO feature-based selector
with mock embeddings or real precomputed features.

Usage:
    # Run with pytest (mock embeddings)
    pytest tests/test_dino_selector_debug.py -v -s

    # Run with real scene (requires precomputed DINO features)
    python tests/test_dino_selector_debug.py --scene /path/to/scene/dslr

    # Run specific test
    pytest tests/test_dino_selector_debug.py::TestDINOSelectorDebug::test_distance_to_selected -v -s
"""

import pytest
import numpy as np
import torch
import sys
import os
import tempfile
from typing import List, Dict
from unittest.mock import Mock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from view_selection.dino_selector import DINOSelector
    DINO_AVAILABLE = True
except ImportError:
    DINO_AVAILABLE = False


# =============================================================================
# Mock Camera Class
# =============================================================================

class MockCamera:
    """Mock camera object matching the real Camera interface."""

    def __init__(
        self,
        uid: int,
        position: np.ndarray = None,
        image_name: str = None,
        image_path: str = None
    ):
        self.uid = uid
        self.image_name = image_name or f"image_{uid:04d}.jpg"
        self.image_path = image_path or f"/fake/path/{self.image_name}"
        if position is None:
            position = np.array([0.0, 0.0, 0.0])
        self.camera_center = torch.tensor(position, dtype=torch.float32)
        self.R = np.eye(3, dtype=np.float32)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def cameras_with_duplicates():
    """
    Create cameras with duplicates.
    Duplicates will have identical DINO embeddings.
    """
    cameras = []
    n_unique = 8
    
    for i in range(n_unique):
        angle = 2 * np.pi * i / n_unique
        pos = np.array([5 * np.cos(angle), 5 * np.sin(angle), 1.0])
        cameras.append(MockCamera(uid=i, position=pos))
    
    # Duplicates of camera 0
    original_pos = cameras[0].camera_center.numpy()
    for dup_idx in range(2):
        dup_cam = MockCamera(
            uid=n_unique + dup_idx,
            position=original_pos.copy(),
            image_name=f"image_0000__dup{dup_idx + 1}.jpg"
        )
        cameras.append(dup_cam)
    
    return cameras


@pytest.fixture
def mock_embeddings_file(cameras_with_duplicates, tmp_path):
    """
    Create a temporary embeddings file with mock DINO features.
    
    Key: Duplicates have IDENTICAL embeddings (same visual content).
    """
    cameras = cameras_with_duplicates
    embedding_dim = 768  # DINOv2 ViT-B/14
    
    embeddings = {}
    
    # Generate unique embeddings for each unique camera position
    np.random.seed(42)
    unique_embeddings = {}
    
    for cam in cameras:
        # Extract base name (without extension)
        base_name = os.path.splitext(cam.image_name)[0]
        
        if "__dup" in cam.image_name:
            # Duplicates get the SAME embedding as original
            original_base = base_name.split("__dup")[0]
            if original_base not in unique_embeddings:
                unique_embeddings[original_base] = np.random.randn(embedding_dim).astype(np.float32)
            embeddings[base_name] = torch.tensor(unique_embeddings[original_base])
        else:
            if base_name not in unique_embeddings:
                unique_embeddings[base_name] = np.random.randn(embedding_dim).astype(np.float32)
            embeddings[base_name] = torch.tensor(unique_embeddings[base_name])
    
    # Add metadata
    embeddings['_model'] = 'mock_dinov2_vitb14'
    embeddings['_embedding_dim'] = embedding_dim
    
    # Save to temp file
    embeddings_path = tmp_path / "features.pt"
    torch.save(embeddings, embeddings_path)
    
    return str(embeddings_path)


@pytest.fixture
def mock_gaussians():
    """Mock Gaussian model."""
    return Mock()


# =============================================================================
# Debug Tests
# =============================================================================

@pytest.mark.skipif(not DINO_AVAILABLE, reason="DINOSelector not available")
class TestDINOSelectorDebug:
    """Debug tests for DINOSelector."""

    def test_distance_to_centroid(self, cameras_with_duplicates, mock_embeddings_file, mock_gaussians):
        """
        Test 'distance_to_centroid' mode: cameras far from feature centroid get higher prob.
        """
        cameras = cameras_with_duplicates
        print(f"\n{'='*60}")
        print("DINO SELECTOR DEBUG - DISTANCE TO CENTROID MODE")
        print(f"{'='*60}")

        config = {
            "embeddings_path": mock_embeddings_file,
            "require_embeddings": True,
            "temperature": 1.0,
            "diversity_mode": "distance_to_centroid",
            "normalize_embeddings": True,
        }
        selector = DINOSelector(config=config, verbose=True, seed=42)

        # --- Step 1: Initialize ---
        print(f"\n[Step 1] Initializing with {len(cameras)} cameras...")
        selector.initialize(cameras)
        
        print(f"  Loaded {len(selector.embeddings)} embeddings")
        print(f"  Embedding dim: {selector.embedding_matrix.shape[1] if selector.embedding_matrix is not None else 'N/A'}")

        # --- Step 2: Compute probabilities ---
        print(f"\n[Step 2] Computing probabilities:")
        probs = selector.compute_probabilities(mock_gaussians, iteration=0)
        
        # Group by base name to show duplicate handling
        print(f"\n  Probabilities (static mode):")
        duplicate_uids = [cam.uid for cam in cameras if "__dup" in cam.image_name]
        
        for cam in cameras:
            prob = probs[cam.uid]
            is_dup = cam.uid in duplicate_uids or cam.uid == 0
            marker = " [DUP/ORIG]" if is_dup else ""
            print(f"    Camera {cam.uid:2d} ({cam.image_name:25s}): prob={prob:.4f}{marker}")

        # Verify sum
        prob_sum = sum(probs.values())
        print(f"\n  Probability sum: {prob_sum:.6f}")
        assert abs(prob_sum - 1.0) < 1e-6
        print("  ✓ Probabilities sum to 1.0")

        # --- Step 3: Check duplicates have same probability ---
        print(f"\n[Step 3] Checking duplicate embeddings:")
        
        orig_prob = probs[0]
        dup_probs = [probs[uid] for uid in duplicate_uids]
        
        print(f"  Camera 0 (original): prob={orig_prob:.4f}")
        for uid in duplicate_uids:
            print(f"  Camera {uid} (duplicate): prob={probs[uid]:.4f}")
        
        # Since duplicates have identical embeddings, they should have same probability
        if all(abs(p - orig_prob) < 0.01 for p in dup_probs):
            print("  ✓ Duplicates have same probability as original (same embedding)")
        else:
            print("  ✗ Duplicates should have same probability!")

        print(f"\n{'='*60}")

    def test_distance_to_selected(self, cameras_with_duplicates, mock_embeddings_file, mock_gaussians):
        """
        Test 'distance_to_selected' mode: cameras far from recently selected get higher prob.
        
        Key check: After selecting a camera, its duplicates should also have LOWER prob.
        """
        cameras = cameras_with_duplicates
        print(f"\n{'='*60}")
        print("DINO SELECTOR DEBUG - DISTANCE TO SELECTED MODE")
        print(f"{'='*60}")

        config = {
            "embeddings_path": mock_embeddings_file,
            "require_embeddings": True,
            "temperature": 0.5,
            "diversity_mode": "distance_to_selected",
            "recency_window": 5,
            "normalize_embeddings": True,
            "use_cumulative_penalty": False,
        }
        selector = DINOSelector(config=config, verbose=True, seed=42)
        selector.initialize(cameras)

        duplicate_uids = [cam.uid for cam in cameras if "__dup" in cam.image_name]

        # --- Step 1: Initial probabilities ---
        print(f"\n[Step 1] Initial probabilities (no prior selections):")
        probs_initial = selector.compute_probabilities(mock_gaussians, iteration=0)
        
        for cam in cameras:
            prob = probs_initial[cam.uid]
            is_dup = cam.uid in duplicate_uids or cam.uid == 0
            marker = " [DUP/ORIG]" if is_dup else ""
            print(f"    Camera {cam.uid:2d}: prob={prob:.4f}{marker}")

        # --- Step 2: Select camera 0 and check probability changes ---
        print(f"\n[Step 2] Selecting Camera 0 and checking changes:")
        
        # Manually add to recent selections
        selector.recent_selections.append(0)
        
        probs_after = selector.compute_probabilities(mock_gaussians, iteration=1)
        
        print(f"\n  Probability changes after selecting Camera 0:")
        for cam in cameras:
            before = probs_initial[cam.uid]
            after = probs_after[cam.uid]
            change = after - before
            marker = "↓" if change < -0.01 else ("↑" if change > 0.01 else "=")
            is_dup = cam.uid in duplicate_uids or cam.uid == 0
            dup_marker = " [DUP/ORIG]" if is_dup else ""
            print(f"    Camera {cam.uid:2d}: {before:.4f} -> {after:.4f} ({marker}){dup_marker}")

        # --- Step 3: Verify duplicates are also penalized ---
        print(f"\n[Step 3] Checking duplicate penalty:")
        
        # Camera 0 was selected, so its duplicates (same embedding) should also be penalized
        orig_prob = probs_after[0]
        dup_probs = [probs_after[uid] for uid in duplicate_uids]
        
        print(f"  Camera 0 (selected): prob={orig_prob:.4f}")
        for uid in duplicate_uids:
            print(f"  Camera {uid} (duplicate): prob={probs_after[uid]:.4f}")
        
        # Duplicates should have same (lowered) probability since they have same embedding
        if all(abs(p - orig_prob) < 0.01 for p in dup_probs):
            print("  ✓ Duplicates penalized equally (feature-based diversity works!)")
        else:
            print("  ✗ Check distance calculation - duplicates should match original")

        # --- Step 4: Run many iterations ---
        print(f"\n[Step 4] Running 500 iterations:")
        
        selector2 = DINOSelector(config=config, verbose=False, seed=42)
        selector2.initialize(cameras)
        
        selection_counts = {cam.uid: 0 for cam in cameras}
        n_iter = 500
        
        for i in range(n_iter):
            cam = selector2.select_view(mock_gaussians, iteration=i)
            selection_counts[cam.uid] += 1

        print(f"\n  Selection distribution:")
        total_dup_pos = 0
        for cam in cameras:
            count = selection_counts[cam.uid]
            pct = 100 * count / n_iter
            is_dup = cam.uid in duplicate_uids or cam.uid == 0
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
            print("    ✓ DINO selector is AVOIDING duplicates (feature-based diversity works!)")
        elif actual > expected_uniform * 1.2:
            print("    ✗ DINO selector is OVER-selecting duplicates!")
        else:
            print("    ≈ Similar to uniform random")

        print(f"\n{'='*60}")

    def test_recency_window(self, cameras_with_duplicates, mock_embeddings_file, mock_gaussians):
        """Test that recency_window controls feature memory."""
        cameras = cameras_with_duplicates
        print(f"\n{'='*60}")
        print("DINO SELECTOR DEBUG - RECENCY WINDOW")
        print(f"{'='*60}")

        config = {
            "embeddings_path": mock_embeddings_file,
            "require_embeddings": True,
            "temperature": 1.0,
            "diversity_mode": "distance_to_selected",
            "recency_window": 3,  # Small window
            "normalize_embeddings": True,
            "use_cumulative_penalty": False,
        }
        selector = DINOSelector(config=config, verbose=True, seed=42)
        selector.initialize(cameras)

        print(f"\n  Recency window: {selector.recency_window}")

        # Select camera 0 three times, then other cameras
        print(f"\n  Simulating selections: 0, 0, 0, 1, 2, 3, 4")
        
        for uid in [0, 0, 0, 1, 2, 3, 4]:
            idx = selector.cam_idx_to_row.get(uid)
            if idx is not None:
                selector.recent_selections.append(idx)
                if len(selector.recent_selections) > selector.recency_window:
                    selector.recent_selections = selector.recent_selections[-selector.recency_window:]
            print(f"    After selecting {uid}: recent (row indices) = {selector.recent_selections}")

        probs = selector.compute_probabilities(mock_gaussians, iteration=7)
        
        print(f"\n  Probabilities after window slides past camera 0:")
        for cam in cameras[:6]:
            print(f"    Camera {cam.uid}: prob={probs[cam.uid]:.4f}")

        print(f"\n  Note: Camera 0's old selections should be 'forgotten'")

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
    """Run DINO selector debug with a real scene."""
    from collections import defaultdict
    
    print(f"\n{'='*60}")
    print(f"DINO SELECTOR - REAL SCENE: {scene_path}")
    print(f"{'='*60}")

    try:
        cameras = load_cameras_from_scene(scene_path)
        print(f"Loaded {len(cameras)} cameras")
    except Exception as e:
        print(f"Failed to load scene: {e}")
        return

    # Look for precomputed embeddings
    embeddings_path = os.path.join(scene_path, "dino_features", "features.pt")
    if not os.path.exists(embeddings_path):
        print(f"ERROR: No precomputed DINO features found at {embeddings_path}")
        print("Run: python extract_dinov3_features.py --scene_path <scene>")
        return

    config = {
        "embeddings_path": embeddings_path,
        "require_embeddings": True,
        "temperature": 0.3,
        "diversity_mode": "distance_to_selected",
        "recency_window": 500,
        "normalize_embeddings": True,
        "use_cumulative_penalty": False,
    }
    
    selector = DINOSelector(config=config, verbose=True, seed=42)
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

    # Check duplicates
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
                print(f"  '{base_name}': {len(group_cams)} cameras")
                print(f"    Probabilities: {[f'{p:.4f}' for p in group_probs]}")
                
                # Check if all have similar probability (should be identical)
                if max(group_probs) - min(group_probs) < 0.01:
                    print(f"    ✓ All duplicates have same probability")
                else:
                    print(f"    ✗ Duplicates have different probabilities!")
    else:
        print(f"  No duplicate cameras found")

    # Run iterations
    print(f"\nRunning 500 iterations...")
    selection_counts = {cam.uid: 0 for cam in cameras}
    n_iter = 500
    
    for i in range(n_iter):
        cam = selector.select_view(mock_gaussians, iteration=i)
        selection_counts[cam.uid] += 1

    # Analyze duplicate position selections
    if duplicate_cameras:
        print(f"\nSelection distribution at duplicated positions:")
        for base_name, group_cams in dup_groups.items():
            if len(group_cams) > 1:
                total = sum(selection_counts[c.uid] for c in group_cams)
                pct = 100 * total / n_iter
                expected = 100 * len(group_cams) / len(cameras)
                print(f"  '{base_name}': {total}/{n_iter} = {pct:.1f}% (expected uniform: {expected:.1f}%)")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug DINOSelector")
    parser.add_argument("--scene", type=str, help="Path to a real scene directory")
    parser.add_argument("--pytest", action="store_true", help="Run pytest instead")
    args = parser.parse_args()

    if not DINO_AVAILABLE:
        print("ERROR: DINOSelector not available. Check imports.")
        sys.exit(1)

    if args.pytest:
        pytest.main([__file__, "-v", "-s"])
    elif args.scene:
        run_debug_with_real_scene(args.scene)
    else:
        print("Running with mock embeddings (use --scene /path/to/scene for real data)")
        pytest.main([__file__, "-v", "-s"])
