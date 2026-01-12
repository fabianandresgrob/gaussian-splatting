"""
Debug test for ScheduledHybridSelector.

This test allows step-by-step debugging of the hybrid selector that combines
multiple strategies with time-dependent weights.

Usage:
    # Run with pytest
    pytest tests/test_hybrid_selector_debug.py -v -s

    # Run specific test
    pytest tests/test_hybrid_selector_debug.py::TestHybridSelectorDebug::test_linear_schedule -v -s
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

from view_selection.scheduled_hybrid_selector import ScheduledHybridSelector, STANDARD_CONFIGS


# =============================================================================
# Mock Camera Class
# =============================================================================

class MockCamera:
    """Mock camera object matching the real Camera interface."""

    def __init__(
        self,
        uid: int,
        position: np.ndarray = None,
        rotation: np.ndarray = None,
        image_name: str = None
    ):
        self.uid = uid
        self.image_name = image_name or f"image_{uid:04d}.jpg"
        if position is None:
            position = np.array([0.0, 0.0, 0.0])
        self.camera_center = torch.tensor(position, dtype=torch.float32)
        if rotation is None:
            self.R = np.eye(3, dtype=np.float32)
        else:
            self.R = rotation.astype(np.float32)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def cameras_simple():
    """Create simple camera list with varied positions."""
    cameras = []
    for i in range(10):
        angle = 2 * np.pi * i / 10
        pos = np.array([5 * np.cos(angle), 5 * np.sin(angle), 1.0])
        cameras.append(MockCamera(uid=i, position=pos))
    return cameras


@pytest.fixture
def cameras_with_duplicates():
    """Create cameras with duplicates."""
    cameras = []
    for i in range(8):
        angle = 2 * np.pi * i / 8
        pos = np.array([5 * np.cos(angle), 5 * np.sin(angle), 1.0])
        cameras.append(MockCamera(uid=i, position=pos))
    
    # Duplicates
    original_pos = cameras[0].camera_center.numpy()
    for dup_idx in range(2):
        dup_cam = MockCamera(
            uid=8 + dup_idx,
            position=original_pos.copy(),
            image_name=f"image_0000__dup{dup_idx + 1}.jpg"
        )
        cameras.append(dup_cam)
    
    return cameras


@pytest.fixture
def mock_gaussians():
    """Mock Gaussian model."""
    return Mock()


# =============================================================================
# Debug Tests
# =============================================================================

class TestHybridSelectorDebug:
    """Debug tests for ScheduledHybridSelector."""

    def test_linear_schedule(self, cameras_simple, mock_gaussians):
        """
        Test linear schedule: weights interpolate from start to end.
        
        Example: [0.8, 0.2] → [0.2, 0.8] over 30k iterations
        """
        cameras = cameras_simple
        print(f"\n{'='*60}")
        print("HYBRID SELECTOR DEBUG - LINEAR SCHEDULE")
        print(f"{'='*60}")

        config = {
            "selectors": ["geometric", "loss_based"],
            "schedule_type": "linear",
            "weights_start": [0.8, 0.2],
            "weights_end": [0.2, 0.8],
            "max_iterations": 30000,
            "temperature": 1.0,
            "geometric_config": {
                "mode": "static",
                "temperature": 1.0,
            },
            "loss_based_config": {
                "temperature": 1.0,
                "min_samples_before_bias": 2,
            }
        }
        selector = ScheduledHybridSelector(config=config, verbose=True, seed=42)
        selector.initialize(cameras)

        print(f"\n  Schedule: {config['weights_start']} → {config['weights_end']}")
        print(f"  Max iterations: {config['max_iterations']}")

        # --- Test weight interpolation at various points ---
        print(f"\n[Step 1] Weight interpolation:")
        
        test_iterations = [0, 7500, 15000, 22500, 30000, 35000]
        
        for it in test_iterations:
            weights = selector._get_current_weights(it)
            print(f"    Iteration {it:6d}: weights = [{weights[0]:.3f}, {weights[1]:.3f}]")

        # Verify endpoints
        weights_0 = selector._get_current_weights(0)
        weights_end = selector._get_current_weights(30000)
        
        if np.allclose(weights_0, [0.8, 0.2], atol=0.01):
            print("\n  ✓ Start weights correct")
        if np.allclose(weights_end, [0.2, 0.8], atol=0.01):
            print("  ✓ End weights correct")

        # --- Test sub-selector contribution ---
        print(f"\n[Step 2] Sub-selector contributions:")
        
        # Early iteration (geometric-heavy)
        probs_early = selector.compute_probabilities(mock_gaussians, iteration=1000)
        
        # Late iteration (loss-heavy)
        # First update losses
        for cam in cameras:
            selector.sub_selectors[1].update_loss(cam, 0.1 + 0.05 * cam.uid)
            selector.sub_selectors[1].update_loss(cam, 0.1 + 0.05 * cam.uid)
        
        probs_late = selector.compute_probabilities(mock_gaussians, iteration=29000)
        
        print(f"\n  Early (it=1000, geo-heavy):")
        sorted_early = sorted(probs_early.items(), key=lambda x: x[1], reverse=True)[:5]
        for uid, p in sorted_early:
            print(f"    Camera {uid}: {p:.4f}")
        
        print(f"\n  Late (it=29000, loss-heavy):")
        sorted_late = sorted(probs_late.items(), key=lambda x: x[1], reverse=True)[:5]
        for uid, p in sorted_late:
            print(f"    Camera {uid}: {p:.4f}")

        print(f"\n{'='*60}")

    def test_step_schedule(self, cameras_simple, mock_gaussians):
        """
        Test step schedule: discrete weight changes at milestones.
        """
        cameras = cameras_simple
        print(f"\n{'='*60}")
        print("HYBRID SELECTOR DEBUG - STEP SCHEDULE")
        print(f"{'='*60}")

        config = {
            "selectors": ["clustering", "geometric", "loss_based"],
            "schedule_type": "step",
            "milestones": [
                [0,     [0.6, 0.3, 0.1]],   # Phase 1: Clustering heavy
                [10000, [0.3, 0.4, 0.3]],   # Phase 2: Balanced
                [20000, [0.1, 0.2, 0.7]],   # Phase 3: Loss heavy
            ],
            "temperature": 1.0,
            "clustering_config": {
                "clustering_method": "kmeans",
                "n_clusters": 4,
            },
            "geometric_config": {
                "mode": "static",
            },
            "loss_based_config": {
                "min_samples_before_bias": 2,
            }
        }
        selector = ScheduledHybridSelector(config=config, verbose=True, seed=42)
        selector.initialize(cameras)

        print(f"\n  Milestones:")
        for it, weights in config['milestones']:
            print(f"    {it:6d}: {weights}")

        # --- Test weight steps ---
        print(f"\n[Step 1] Weight at various iterations:")
        
        test_iterations = [0, 5000, 10000, 15000, 20000, 25000]
        
        for it in test_iterations:
            weights = selector._get_current_weights(it)
            print(f"    Iteration {it:6d}: [{weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f}]")

        # Verify step behavior
        w_5000 = selector._get_current_weights(5000)
        w_10000 = selector._get_current_weights(10000)
        
        if np.allclose(w_5000, [0.6, 0.3, 0.1]):
            print("\n  ✓ Before first milestone: correct weights")
        if np.allclose(w_10000, [0.3, 0.4, 0.3]):
            print("  ✓ At second milestone: weights stepped correctly")

        print(f"\n{'='*60}")

    def test_cosine_schedule(self, cameras_simple, mock_gaussians):
        """
        Test cosine annealing schedule: smooth S-curve transition.
        """
        cameras = cameras_simple
        print(f"\n{'='*60}")
        print("HYBRID SELECTOR DEBUG - COSINE SCHEDULE")
        print(f"{'='*60}")

        config = {
            "selectors": ["geometric", "loss_based"],
            "schedule_type": "cosine",
            "weights_start": [0.9, 0.1],
            "weights_end": [0.1, 0.9],
            "max_iterations": 30000,
            "temperature": 1.0,
            "geometric_config": {"mode": "static"},
            "loss_based_config": {"min_samples_before_bias": 2}
        }
        selector = ScheduledHybridSelector(config=config, verbose=True, seed=42)
        selector.initialize(cameras)

        print(f"\n  Cosine schedule: {config['weights_start']} → {config['weights_end']}")

        # --- Show cosine curve ---
        print(f"\n[Step 1] Weight trajectory (cosine annealing):")
        
        iterations = [0, 3000, 7500, 15000, 22500, 27000, 30000]
        
        for it in iterations:
            weights = selector._get_current_weights(it)
            progress = it / 30000
            # Cosine should be slow at start/end, fast in middle
            print(f"    Iteration {it:6d} ({progress*100:5.1f}%): "
                  f"[{weights[0]:.3f}, {weights[1]:.3f}]")

        # Compare linear vs cosine at midpoint
        print(f"\n  At midpoint (15000):")
        print(f"    Cosine: {selector._get_current_weights(15000)}")
        print(f"    Linear would be: [0.5, 0.5]")
        print(f"    (Cosine should also be ~[0.5, 0.5] at midpoint)")

        print(f"\n{'='*60}")

    def test_standard_presets(self, cameras_simple, mock_gaussians):
        """Test the standard preset configurations."""
        cameras = cameras_simple
        print(f"\n{'='*60}")
        print("HYBRID SELECTOR DEBUG - STANDARD PRESETS")
        print(f"{'='*60}")

        print(f"\n  Available presets: {list(STANDARD_CONFIGS.keys())}")

        for preset_name in ['explore_then_exploit', 'three_phase_training']:
            print(f"\n  --- Preset: {preset_name} ---")
            
            preset_config = STANDARD_CONFIGS[preset_name]
            print(f"    Description: {preset_config.get('description', 'N/A')}")
            print(f"    Selectors: {preset_config['selectors']}")
            print(f"    Schedule: {preset_config['schedule_type']}")

            config = {"preset": preset_name}
            selector = ScheduledHybridSelector(config=config, verbose=False, seed=42)
            selector.initialize(cameras)

            # Show weights at key points
            print(f"    Weights over training:")
            for it in [0, 10000, 20000, 30000]:
                weights = selector._get_current_weights(it)
                weights_str = ", ".join(f"{w:.2f}" for w in weights)
                print(f"      {it:6d}: [{weights_str}]")

        print(f"\n{'='*60}")

    def test_duplicate_handling(self, cameras_with_duplicates, mock_gaussians):
        """
        Test how hybrid selector handles duplicates.
        
        Depends on sub-selectors: geometric should penalize duplicates,
        loss_based will not (tracks by uid).
        """
        cameras = cameras_with_duplicates
        print(f"\n{'='*60}")
        print("HYBRID SELECTOR DEBUG - DUPLICATE HANDLING")
        print(f"{'='*60}")

        duplicate_uids = [cam.uid for cam in cameras if "__dup" in cam.image_name]
        
        # Test with geometric-heavy config (should avoid duplicates)
        config_geo = {
            "selectors": ["geometric", "loss_based"],
            "schedule_type": "linear",
            "weights_start": [0.9, 0.1],
            "weights_end": [0.9, 0.1],  # Stay geometric
            "max_iterations": 30000,
            "geometric_config": {
                "mode": "distance_to_selected",
                "recency_window": 50,
                "use_cumulative_penalty": False,
            },
            "loss_based_config": {"min_samples_before_bias": 2}
        }
        
        # Test with loss-heavy config (will not avoid duplicates)
        config_loss = {
            "selectors": ["geometric", "loss_based"],
            "schedule_type": "linear",
            "weights_start": [0.1, 0.9],
            "weights_end": [0.1, 0.9],  # Stay loss-heavy
            "max_iterations": 30000,
            "geometric_config": {"mode": "static"},
            "loss_based_config": {"min_samples_before_bias": 2}
        }

        results = {}
        
        for name, config in [("Geometric-heavy", config_geo), ("Loss-heavy", config_loss)]:
            print(f"\n  --- {name} ---")
            
            selector = ScheduledHybridSelector(config=config, verbose=False, seed=42)
            selector.initialize(cameras)
            
            # Give all cameras similar loss
            for cam in cameras:
                for _ in range(5):
                    selector.sub_selectors[1].update_loss(cam, 0.2)
            
            # Run iterations
            selection_counts = {cam.uid: 0 for cam in cameras}
            n_iter = 500
            
            for i in range(n_iter):
                cam = selector.select_view(mock_gaussians, iteration=i)
                selection_counts[cam.uid] += 1
            
            # Count selections at duplicated position
            dup_total = sum(selection_counts[uid] for uid in [0] + duplicate_uids)
            pct = 100 * dup_total / n_iter
            results[name] = pct
            
            print(f"    Selections at duplicated position: {dup_total}/{n_iter} = {pct:.1f}%")

        expected = 100 * (len(duplicate_uids) + 1) / len(cameras)
        print(f"\n  Expected if uniform: {expected:.1f}%")
        
        if results["Geometric-heavy"] < results["Loss-heavy"] * 0.8:
            print("  ✓ Geometric-heavy config avoids duplicates better")
        else:
            print("  ≈ Both configs similar (check geometric sub-selector)")

        print(f"\n{'='*60}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug ScheduledHybridSelector")
    parser.add_argument("--pytest", action="store_true", help="Run pytest")
    args = parser.parse_args()

    pytest.main([__file__, "-v", "-s"])
