"""
Debug tests for baseline selectors: Stack, UniformRandom, Sequential.

These are simple selectors that serve as baselines for comparison.

Usage:
    # Run with pytest
    pytest tests/test_baseline_selectors_debug.py -v -s

    # Run specific test
    pytest tests/test_baseline_selectors_debug.py::TestStackSelectorDebug -v -s
"""

import pytest
import numpy as np
import torch
import sys
import os
from typing import List, Dict
from unittest.mock import Mock
from collections import Counter

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from view_selection.stack_selector import StackBasedSelector
from view_selection.random_selector import UniformRandomSelector
from view_selection.sequential_selector import SequentialSelector


# =============================================================================
# Mock Camera Class
# =============================================================================

class MockCamera:
    """Mock camera object matching the real Camera interface."""

    def __init__(self, uid: int, image_name: str = None):
        self.uid = uid
        self.image_name = image_name or f"image_{uid:04d}.jpg"
        self.camera_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        self.R = np.eye(3, dtype=np.float32)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def cameras_simple():
    """Create simple camera list."""
    return [MockCamera(uid=i) for i in range(10)]


@pytest.fixture
def cameras_with_duplicates():
    """Create cameras with duplicates (same content, different uid)."""
    cameras = []
    for i in range(8):
        cameras.append(MockCamera(uid=i))
    
    # Add duplicates
    for dup_idx in range(2):
        dup_cam = MockCamera(
            uid=8 + dup_idx,
            image_name=f"image_0000__dup{dup_idx + 1}.jpg"
        )
        cameras.append(dup_cam)
    
    return cameras


@pytest.fixture
def mock_gaussians():
    """Mock Gaussian model."""
    return Mock()


# =============================================================================
# Stack Selector Tests
# =============================================================================

class TestStackSelectorDebug:
    """
    Debug tests for StackBasedSelector.
    
    This is the original 3DGS behavior: shuffle cameras into a stack,
    pop one at a time. When stack is empty, reshuffle.
    
    Key property: Each camera seen exactly once per epoch (no replacement).
    """

    def test_stack_basic(self, cameras_simple, mock_gaussians):
        """Test basic stack behavior: each camera seen once per epoch."""
        cameras = cameras_simple
        print(f"\n{'='*60}")
        print("STACK SELECTOR DEBUG - BASIC BEHAVIOR")
        print(f"{'='*60}")

        selector = StackBasedSelector(config={}, verbose=True, seed=42)
        selector.initialize(cameras)

        n_cameras = len(cameras)
        print(f"\n  Initialized with {n_cameras} cameras")

        # --- Test 1: First epoch (all cameras exactly once) ---
        print(f"\n[Test 1] First epoch ({n_cameras} selections):")
        
        first_epoch = []
        for i in range(n_cameras):
            cam = selector.select_view(mock_gaussians, iteration=i)
            first_epoch.append(cam.uid)
            print(f"    Selection {i+1}: Camera {cam.uid}")

        unique_first = set(first_epoch)
        print(f"\n  Unique cameras selected: {len(unique_first)}/{n_cameras}")
        
        if len(unique_first) == n_cameras:
            print("  [OK] Each camera selected exactly once in first epoch")
        else:
            print("  [FAIL] Some cameras missed or duplicated!")

        # --- Test 2: Second epoch (reshuffled) ---
        print(f"\n[Test 2] Second epoch (should be reshuffled):")
        
        second_epoch = []
        for i in range(n_cameras):
            cam = selector.select_view(mock_gaussians, iteration=n_cameras + i)
            second_epoch.append(cam.uid)

        print(f"  First epoch order:  {first_epoch}")
        print(f"  Second epoch order: {second_epoch}")
        
        unique_second = set(second_epoch)
        if len(unique_second) == n_cameras:
            print("  [OK] Each camera selected exactly once in second epoch")
        
        if first_epoch != second_epoch:
            print("  [OK] Order is different (reshuffled)")
        else:
            print("  ~ Same order (possible but unlikely)")

        print(f"\n{'='*60}")

    def test_stack_with_duplicates(self, cameras_with_duplicates, mock_gaussians):
        """
        Test stack behavior with duplicate cameras.
        
        Key insight: Stack treats each camera independently by uid.
        Duplicates will be selected just as often as unique cameras.
        """
        cameras = cameras_with_duplicates
        print(f"\n{'='*60}")
        print("STACK SELECTOR DEBUG - DUPLICATE HANDLING")
        print(f"{'='*60}")

        selector = StackBasedSelector(config={}, verbose=True, seed=42)
        selector.initialize(cameras)

        n_cameras = len(cameras)
        duplicate_uids = [cam.uid for cam in cameras if "__dup" in cam.image_name]
        
        print(f"\n  Total cameras: {n_cameras}")
        print(f"  Duplicate camera uids: {duplicate_uids}")

        # Run multiple epochs
        n_epochs = 5
        selection_counts = {cam.uid: 0 for cam in cameras}
        
        for epoch in range(n_epochs):
            for i in range(n_cameras):
                cam = selector.select_view(mock_gaussians, iteration=epoch * n_cameras + i)
                selection_counts[cam.uid] += 1

        print(f"\n  Selection counts after {n_epochs} epochs:")
        total_dup = 0
        for cam in cameras:
            count = selection_counts[cam.uid]
            is_dup = cam.uid in duplicate_uids
            marker = " [DUP]" if is_dup else ""
            print(f"    Camera {cam.uid}: {count}{marker}")
            if is_dup:
                total_dup += count

        expected_per_cam = n_epochs
        print(f"\n  Expected per camera: {expected_per_cam}")
        print(f"  Total selections at duplicated position: {total_dup}/{n_epochs * n_cameras}")
        
        # All cameras should have same count (no position awareness)
        all_equal = all(c == expected_per_cam for c in selection_counts.values())
        if all_equal:
            print("\n  [OK] All cameras selected equally (no duplicate awareness)")
        
        print(f"\n  NOTE: Stack selector has NO duplicate awareness.")
        print(f"        Each camera (by uid) is treated independently.")

        print(f"\n{'='*60}")


# =============================================================================
# Uniform Random Selector Tests
# =============================================================================

class TestUniformRandomSelectorDebug:
    """
    Debug tests for UniformRandomSelector.
    
    True uniform random sampling WITH replacement.
    Each camera has equal probability at every selection.
    """

    def test_uniform_distribution(self, cameras_simple, mock_gaussians):
        """Test that selections are uniformly distributed."""
        cameras = cameras_simple
        print(f"\n{'='*60}")
        print("UNIFORM RANDOM SELECTOR DEBUG - DISTRIBUTION")
        print(f"{'='*60}")

        selector = UniformRandomSelector(config={}, verbose=True, seed=42)
        selector.initialize(cameras)

        n_cameras = len(cameras)
        n_iter = 1000

        print(f"\n  Cameras: {n_cameras}")
        print(f"  Iterations: {n_iter}")
        print(f"  Expected per camera: {n_iter / n_cameras:.1f}")

        # Run many iterations
        selection_counts = {cam.uid: 0 for cam in cameras}
        for i in range(n_iter):
            cam = selector.select_view(mock_gaussians, iteration=i)
            selection_counts[cam.uid] += 1

        print(f"\n  Selection counts:")
        expected = n_iter / n_cameras
        max_deviation = 0
        for cam in cameras:
            count = selection_counts[cam.uid]
            deviation = abs(count - expected) / expected * 100
            max_deviation = max(max_deviation, deviation)
            print(f"    Camera {cam.uid}: {count} ({deviation:.1f}% from expected)")

        print(f"\n  Max deviation from expected: {max_deviation:.1f}%")
        
        # With 1000 iterations and 10 cameras, expect ~100 each
        # Reasonable variance should keep deviation under ~30%
        if max_deviation < 30:
            print("  [OK] Distribution looks uniform (within expected variance)")
        else:
            print("  [WARN] High deviation - might want to check randomness")

        # Test probabilities
        probs = selector.compute_probabilities(mock_gaussians, iteration=0)
        print(f"\n  Probabilities: {list(probs.values())[:5]}...")
        
        is_uniform = all(abs(p - 1.0/n_cameras) < 1e-6 for p in probs.values())
        if is_uniform:
            print("  [OK] Probabilities are exactly uniform")

        print(f"\n{'='*60}")

    def test_duplicate_handling(self, cameras_with_duplicates, mock_gaussians):
        """
        Test uniform random with duplicates.
        
        Same as stack: no position awareness, each uid equally likely.
        """
        cameras = cameras_with_duplicates
        print(f"\n{'='*60}")
        print("UNIFORM RANDOM SELECTOR DEBUG - DUPLICATES")
        print(f"{'='*60}")

        selector = UniformRandomSelector(config={}, verbose=True, seed=42)
        selector.initialize(cameras)

        n_cameras = len(cameras)
        duplicate_uids = [cam.uid for cam in cameras if "__dup" in cam.image_name]
        
        n_iter = 1000
        selection_counts = {cam.uid: 0 for cam in cameras}
        
        for i in range(n_iter):
            cam = selector.select_view(mock_gaussians, iteration=i)
            selection_counts[cam.uid] += 1

        print(f"\n  Selection counts after {n_iter} iterations:")
        total_dup = 0
        for cam in cameras:
            count = selection_counts[cam.uid]
            pct = 100 * count / n_iter
            is_dup = cam.uid in duplicate_uids
            marker = " [DUP]" if is_dup else ""
            print(f"    Camera {cam.uid}: {count} ({pct:.1f}%){marker}")
            if is_dup:
                total_dup += count

        # Duplicates should get ~20% of selections (2 out of 10 cameras)
        expected_dup_pct = 100 * len(duplicate_uids) / n_cameras
        actual_dup_pct = 100 * total_dup / n_iter
        
        print(f"\n  Duplicate cameras: {len(duplicate_uids)}/{n_cameras}")
        print(f"  Expected selections: {expected_dup_pct:.1f}%")
        print(f"  Actual selections: {actual_dup_pct:.1f}%")
        
        print(f"\n  NOTE: Uniform random has NO duplicate awareness.")

        print(f"\n{'='*60}")


# =============================================================================
# Sequential Selector Tests
# =============================================================================

class TestSequentialSelectorDebug:
    """
    Debug tests for SequentialSelector.
    
    Deterministic: cameras selected in fixed order (by uid), wrapping around.
    No randomness at all.
    """

    def test_sequential_order(self, cameras_simple, mock_gaussians):
        """Test that cameras are selected in strict sequential order."""
        cameras = cameras_simple
        print(f"\n{'='*60}")
        print("SEQUENTIAL SELECTOR DEBUG - ORDER")
        print(f"{'='*60}")

        selector = SequentialSelector(config={}, verbose=True, seed=42)
        selector.initialize(cameras)

        n_cameras = len(cameras)
        print(f"\n  Cameras: {n_cameras}")
        print(f"  Camera order (by uid): {[cam.uid for cam in selector.camera_order]}")

        # Select 2.5 epochs worth
        n_selections = int(n_cameras * 2.5)
        selections = []
        
        print(f"\n  Selections ({n_selections} total):")
        for i in range(n_selections):
            cam = selector.select_view(mock_gaussians, iteration=i)
            selections.append(cam.uid)
        
        # Show first two epochs
        print(f"    First epoch:  {selections[:n_cameras]}")
        print(f"    Second epoch: {selections[n_cameras:2*n_cameras]}")
        print(f"    Partial third: {selections[2*n_cameras:]}")

        # Verify order is strictly sequential
        expected = list(range(n_cameras)) * 3  # 3 epochs
        expected = expected[:n_selections]
        
        if selections == expected:
            print("\n  [OK] Order is strictly sequential (0, 1, 2, ..., N-1, 0, 1, ...)")
        else:
            print("\n  [FAIL] Order is not sequential!")
            print(f"    Expected: {expected}")
            print(f"    Got: {selections}")

        # Test determinism
        print(f"\n[Test] Determinism check (same seed should give same result):")
        selector2 = SequentialSelector(config={}, verbose=False, seed=42)
        selector2.initialize(cameras)
        
        selections2 = [selector2.select_view(mock_gaussians, i).uid for i in range(n_selections)]
        
        if selections == selections2:
            print("  [OK] Fully deterministic")
        else:
            print("  [FAIL] Not deterministic!")

        print(f"\n{'='*60}")

    def test_probabilities(self, cameras_simple, mock_gaussians):
        """Test that probabilities are always 1.0 for current, 0.0 for others."""
        cameras = cameras_simple
        print(f"\n{'='*60}")
        print("SEQUENTIAL SELECTOR DEBUG - PROBABILITIES")
        print(f"{'='*60}")

        selector = SequentialSelector(config={}, verbose=True, seed=42)
        selector.initialize(cameras)

        print(f"\n  Checking probabilities at each step:")
        
        for i in range(3):
            probs = selector.compute_probabilities(mock_gaussians, iteration=i)
            current = selector.camera_order[selector.current_index].uid
            
            print(f"\n    Iteration {i}: current_index={selector.current_index}, current_uid={current}")
            
            # Verify exactly one camera has prob 1.0
            ones = [uid for uid, p in probs.items() if p == 1.0]
            zeros = [uid for uid, p in probs.items() if p == 0.0]
            
            print(f"      Cameras with prob=1.0: {ones}")
            print(f"      Cameras with prob=0.0: {len(zeros)}")
            
            if len(ones) == 1 and ones[0] == current:
                print(f"      [OK] Correct: only camera {current} has prob=1.0")
            else:
                print(f"      [FAIL] Incorrect probabilities!")
            
            # Make a selection to advance
            selector.select_view(mock_gaussians, iteration=i)

        print(f"\n{'='*60}")


# =============================================================================
# Comparison Test
# =============================================================================

class TestBaselineComparison:
    """Compare all baseline selectors side by side."""

    def test_compare_baselines(self, cameras_with_duplicates, mock_gaussians):
        """Compare selection patterns across all baseline selectors."""
        cameras = cameras_with_duplicates
        print(f"\n{'='*60}")
        print("BASELINE SELECTOR COMPARISON")
        print(f"{'='*60}")

        n_cameras = len(cameras)
        duplicate_uids = [cam.uid for cam in cameras if "__dup" in cam.image_name]
        n_iter = 500

        selectors = {
            "Stack": StackBasedSelector(config={}, verbose=False, seed=42),
            "Uniform": UniformRandomSelector(config={}, verbose=False, seed=42),
            "Sequential": SequentialSelector(config={}, verbose=False, seed=42),
        }

        print(f"\n  Cameras: {n_cameras} (including {len(duplicate_uids)} duplicates)")
        print(f"  Iterations: {n_iter}")

        results = {}
        for name, selector in selectors.items():
            selector.initialize(cameras)
            
            counts = {cam.uid: 0 for cam in cameras}
            for i in range(n_iter):
                cam = selector.select_view(mock_gaussians, iteration=i)
                counts[cam.uid] += 1
            
            results[name] = counts

        # Print comparison
        print(f"\n  Selection counts by camera:")
        print(f"    {'Camera':<10} | {'Stack':>8} | {'Uniform':>8} | {'Sequential':>10}")
        print(f"    {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
        
        for cam in cameras:
            uid = cam.uid
            marker = " [DUP]" if uid in duplicate_uids else ""
            print(f"    {f'Camera {uid}':<10} | "
                  f"{results['Stack'][uid]:>8} | "
                  f"{results['Uniform'][uid]:>8} | "
                  f"{results['Sequential'][uid]:>10}{marker}")

        # Summary: selections at duplicated position
        print(f"\n  Selections at duplicated position:")
        for name in selectors:
            dup_total = sum(results[name][uid] for uid in duplicate_uids)
            pct = 100 * dup_total / n_iter
            print(f"    {name:<12}: {dup_total}/{n_iter} = {pct:.1f}%")

        expected_uniform = 100 * len(duplicate_uids) / n_cameras
        print(f"    Expected (uniform): {expected_uniform:.1f}%")

        print(f"\n  Key insight: None of these baselines have duplicate awareness.")
        print(f"               They all treat each camera (by uid) independently.")

        print(f"\n{'='*60}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug baseline selectors")
    parser.add_argument("--pytest", action="store_true", help="Run pytest")
    args = parser.parse_args()

    pytest.main([__file__, "-v", "-s"])
