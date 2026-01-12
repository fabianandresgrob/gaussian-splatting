"""
Debug test for LossBasedSelector.

This test allows step-by-step debugging of the loss-based selector with
mock cameras and simulated loss updates.

Usage:
    # Run with pytest (mock cameras)
    pytest tests/test_loss_selector_debug.py -v -s

    # Run specific test
    pytest tests/test_loss_selector_debug.py::TestLossSelectorDebug::test_loss_tracking -v -s
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

from view_selection.loss_selector import LossBasedSelector


# =============================================================================
# Mock Camera Class
# =============================================================================

class MockCamera:
    """Mock camera object matching the real Camera interface."""

    def __init__(
        self,
        uid: int,
        position: np.ndarray = None,
        image_name: str = None
    ):
        self.uid = uid
        self.image_name = image_name or f"image_{uid:04d}.jpg"
        if position is None:
            position = np.array([0.0, 0.0, 0.0])
        self.camera_center = torch.tensor(position, dtype=torch.float32)
        self.R = np.eye(3, dtype=np.float32)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def cameras_simple():
    """Create simple camera setup for loss testing."""
    return [MockCamera(uid=i) for i in range(10)]


@pytest.fixture
def cameras_with_duplicates():
    """Create cameras with duplicates at same position."""
    cameras = []
    n_unique = 8
    
    for i in range(n_unique):
        angle = 2 * np.pi * i / n_unique
        pos = np.array([5 * np.cos(angle), 5 * np.sin(angle), 1.0])
        cameras.append(MockCamera(uid=i, position=pos))
    
    # Add duplicates
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
def mock_gaussians():
    """Mock Gaussian model (unused by loss selector directly)."""
    return Mock()


# =============================================================================
# Debug Tests
# =============================================================================

class TestLossSelectorDebug:
    """Debug tests for LossBasedSelector."""

    def test_loss_tracking(self, cameras_simple, mock_gaussians):
        """
        Test that loss values are properly tracked and affect probabilities.
        
        Key behavior:
        - Higher loss → Higher selection probability
        - No EMA smoothing (raw loss stored directly)
        """
        cameras = cameras_simple
        print(f"\n{'='*60}")
        print("LOSS SELECTOR DEBUG - LOSS TRACKING")
        print(f"{'='*60}")

        config = {
            "temperature": 1.0,
            "min_samples_before_bias": 2,
        }
        selector = LossBasedSelector(config=config, verbose=True, seed=42)

        # --- Step 1: Initialize ---
        print(f"\n[Step 1] Initializing with {len(cameras)} cameras...")
        selector.initialize(cameras)
        
        print(f"  Configuration:")
        print(f"    temperature: {selector.temperature}")
        print(f"    min_samples_before_bias: {selector.min_samples_before_bias}")

        # --- Step 2: Check initial state (uniform) ---
        print(f"\n[Step 2] Initial state (no losses recorded):")
        probs = selector.compute_probabilities(mock_gaussians, iteration=0)
        
        print(f"  Initial probabilities (should be uniform):")
        for cam in cameras[:5]:
            print(f"    Camera {cam.uid}: prob={probs[cam.uid]:.4f}, "
                  f"loss={selector.losses[cam.uid]:.4f}, "
                  f"samples={selector.loss_sample_counts[cam.uid]}")
        
        # Verify uniform
        expected = 1.0 / len(cameras)
        is_uniform = all(abs(p - expected) < 1e-6 for p in probs.values())
        if is_uniform:
            print(f"  ✓ Probabilities are uniform ({expected:.4f} each)")
        else:
            print(f"  ✗ Probabilities are NOT uniform!")

        # --- Step 3: Update losses for some cameras ---
        print(f"\n[Step 3] Updating losses:")
        
        # Camera 0: high loss (should get high probability)
        # Camera 1: medium loss
        # Camera 2: low loss (should get low probability)
        loss_values = {
            0: 0.5,   # High
            1: 0.3,   # Medium
            2: 0.1,   # Low
            3: 0.2,   # Medium-low
        }
        
        # Update losses multiple times to exceed min_samples_before_bias
        for _ in range(3):
            for uid, loss in loss_values.items():
                cam = cameras[uid]
                selector.update_loss(cam, loss)
                print(f"  Updated Camera {uid}: loss={loss}")
        
        print(f"\n  Loss tracking state:")
        for cam in cameras:
            print(f"    Camera {cam.uid}: loss={selector.losses[cam.uid]:.4f}, "
                  f"samples={selector.loss_sample_counts[cam.uid]}")

        # --- Step 4: Check probabilities after loss updates ---
        print(f"\n[Step 4] Probabilities after loss updates:")
        probs_after = selector.compute_probabilities(mock_gaussians, iteration=10)
        
        sorted_probs = sorted(probs_after.items(), key=lambda x: x[1], reverse=True)
        for uid, prob in sorted_probs:
            loss = selector.losses[uid]
            samples = selector.loss_sample_counts[uid]
            expected_high = uid == 0  # Camera 0 should have highest prob
            marker = " ← HIGH LOSS" if uid == 0 else (" ← LOW LOSS" if uid == 2 else "")
            print(f"    Camera {uid}: prob={prob:.4f}, loss={loss:.4f}, samples={samples}{marker}")
        
        # Verify high-loss camera has highest probability
        if sorted_probs[0][0] == 0:
            print(f"\n  ✓ Camera 0 (highest loss) has highest probability")
        else:
            print(f"\n  ✗ Camera 0 should have highest probability!")

        # Verify probabilities sum to 1
        prob_sum = sum(probs_after.values())
        print(f"\n  Probability sum: {prob_sum:.6f}")
        assert abs(prob_sum - 1.0) < 1e-6
        print("  ✓ Probabilities sum to 1.0")

        print(f"\n{'='*60}")
        print("DEBUG COMPLETE")
        print(f"{'='*60}")

    def test_min_samples_threshold(self, cameras_simple, mock_gaussians):
        """
        Test that loss bias only kicks in after min_samples_before_bias threshold.
        """
        cameras = cameras_simple
        print(f"\n{'='*60}")
        print("LOSS SELECTOR DEBUG - MIN SAMPLES THRESHOLD")
        print(f"{'='*60}")

        config = {
            "temperature": 1.0,
            "min_samples_before_bias": 5,  # Need 5 samples before bias kicks in
        }
        selector = LossBasedSelector(config=config, verbose=True, seed=42)
        selector.initialize(cameras)

        print(f"\n  min_samples_before_bias: {selector.min_samples_before_bias}")

        # Update camera 0 with high loss, but only 3 times (below threshold)
        print(f"\n[Test 1] Camera 0: 3 samples (below threshold)")
        for _ in range(3):
            selector.update_loss(cameras[0], 0.9)  # Very high loss
        
        probs = selector.compute_probabilities(mock_gaussians, iteration=5)
        print(f"  Camera 0: samples={selector.loss_sample_counts[0]}, "
              f"loss={selector.losses[0]:.4f}, prob={probs[0]:.4f}")
        
        # Should still be uniform since not enough samples
        expected = 1.0 / len(cameras)
        is_uniform = abs(probs[0] - expected) < 0.01
        if is_uniform:
            print(f"  ✓ Probability still ~uniform (bias not active yet)")
        else:
            print(f"  ✗ Probability should be uniform below threshold!")

        # Now update more cameras to exceed threshold
        print(f"\n[Test 2] All cameras get 5+ samples")
        for cam in cameras:
            loss = 0.1 if cam.uid != 0 else 0.9  # Camera 0 high, others low
            for _ in range(5):
                selector.update_loss(cam, loss)
        
        probs_after = selector.compute_probabilities(mock_gaussians, iteration=50)
        
        print(f"\n  After reaching threshold:")
        for cam in cameras[:5]:
            prob = probs_after[cam.uid]
            loss = selector.losses[cam.uid]
            samples = selector.loss_sample_counts[cam.uid]
            print(f"    Camera {cam.uid}: samples={samples}, loss={loss:.4f}, prob={prob:.4f}")
        
        # Now camera 0 should have highest probability
        if probs_after[0] > probs_after[1]:
            print(f"\n  ✓ Camera 0 (high loss) now has higher probability than others")
        else:
            print(f"\n  ✗ Loss bias not working correctly!")

        print(f"\n{'='*60}")

    def test_temperature_effect(self, cameras_simple, mock_gaussians):
        """
        Test how temperature affects probability distribution.
        
        Lower temperature → more peaked distribution (stronger bias)
        Higher temperature → flatter distribution (weaker bias)
        """
        cameras = cameras_simple
        print(f"\n{'='*60}")
        print("LOSS SELECTOR DEBUG - TEMPERATURE EFFECT")
        print(f"{'='*60}")

        # Set up losses: camera 0 high, others low
        base_losses = {0: 0.8, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1}

        for temp in [0.3, 1.0, 2.0]:
            print(f"\n  Temperature = {temp}:")
            
            config = {"temperature": temp, "min_samples_before_bias": 2}
            selector = LossBasedSelector(config=config, verbose=False, seed=42)
            selector.initialize(cameras)
            
            # Update losses
            for cam in cameras:
                loss = base_losses.get(cam.uid, 0.1)
                for _ in range(5):
                    selector.update_loss(cam, loss)
            
            probs = selector.compute_probabilities(mock_gaussians, iteration=10)
            
            cam0_prob = probs[0]
            others_avg = np.mean([probs[i] for i in range(1, 5)])
            ratio = cam0_prob / others_avg if others_avg > 0 else float('inf')
            
            print(f"    Camera 0 (high loss): prob={cam0_prob:.4f}")
            print(f"    Others average: prob={others_avg:.4f}")
            print(f"    Ratio (cam0/others): {ratio:.2f}x")

        print(f"\n{'='*60}")

    def test_duplicate_handling(self, cameras_with_duplicates, mock_gaussians):
        """
        Test how loss selector handles duplicates.
        
        Key insight: Loss selector tracks by camera UID, not position.
        So duplicates are treated as completely independent cameras.
        """
        cameras = cameras_with_duplicates
        print(f"\n{'='*60}")
        print("LOSS SELECTOR DEBUG - DUPLICATE HANDLING")
        print(f"{'='*60}")

        config = {
            "temperature": 0.5,
            "min_samples_before_bias": 2,
        }
        selector = LossBasedSelector(config=config, verbose=True, seed=42)
        selector.initialize(cameras)

        # Find duplicates
        duplicate_uids = [cam.uid for cam in cameras if "__dup" in cam.image_name]
        original_uid = 0
        
        print(f"\n  Original camera: {original_uid}")
        print(f"  Duplicate cameras: {duplicate_uids}")

        # Give all cameras in duplicated position HIGH loss
        # This simulates them all being "hard" views
        print(f"\n[Test] All cameras at duplicated position get HIGH loss:")
        
        for cam in cameras:
            is_at_dup_pos = cam.uid == original_uid or cam.uid in duplicate_uids
            loss = 0.8 if is_at_dup_pos else 0.1
            for _ in range(5):
                selector.update_loss(cam, loss)
            marker = "[DUP/ORIG]" if is_at_dup_pos else ""
            print(f"    Camera {cam.uid}: loss={loss} {marker}")

        probs = selector.compute_probabilities(mock_gaussians, iteration=50)

        # Run many iterations
        print(f"\n  Running 500 selections:")
        selection_counts = {cam.uid: 0 for cam in cameras}
        for i in range(500):
            cam = selector.select_view(mock_gaussians, iteration=i)
            selection_counts[cam.uid] += 1

        total_dup_pos = 0
        for cam in cameras:
            count = selection_counts[cam.uid]
            is_dup = cam.uid == original_uid or cam.uid in duplicate_uids
            marker = " [DUP/ORIG]" if is_dup else ""
            pct = 100 * count / 500
            print(f"    Camera {cam.uid}: {count:4d} ({pct:5.1f}%){marker}")
            if is_dup:
                total_dup_pos += count

        n_dup_cams = len(duplicate_uids) + 1
        expected_uniform = 100 * n_dup_cams / len(cameras)
        actual = 100 * total_dup_pos / 500
        
        print(f"\n  Summary:")
        print(f"    Selections at duplicated position: {total_dup_pos}/500 = {actual:.1f}%")
        print(f"    Expected if uniform: {expected_uniform:.1f}%")
        
        # Loss selector will OVER-select duplicates because:
        # 1. Each duplicate has its own high loss (tracked by UID)
        # 2. No position-based awareness
        print(f"\n  NOTE: Loss selector treats each camera independently by UID.")
        print(f"        It has no concept of 'same position', so duplicates")
        print(f"        with high loss will ALL be selected frequently.")

        print(f"\n{'='*60}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug LossBasedSelector")
    parser.add_argument("--pytest", action="store_true", help="Run pytest")
    args = parser.parse_args()

    if args.pytest:
        pytest.main([__file__, "-v", "-s"])
    else:
        pytest.main([__file__, "-v", "-s"])
