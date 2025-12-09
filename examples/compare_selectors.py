#!/usr/bin/env python3
"""
Example script for comparing different view selection strategies.

This script demonstrates how to:
1. Compare multiple selector strategies
2. Generate comparison visualizations
3. Aggregate statistics across experiments

Usage:
    python compare_selectors.py <experiment_root>

Where experiment_root contains subdirectories like:
    experiment_root/
        garden_random/logs/
        garden_clustering/logs/
        garden_loss_based/logs/
        ...
"""

import sys
import os
import json
import glob

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from view_selection.analysis import (
    load_selection_history,
    compare_strategies,
    export_summary_stats,
)
import pandas as pd


def find_experiment_logs(root_dir: str, pattern: str = "*/logs") -> dict:
    """
    Find all experiment log directories.

    Args:
        root_dir: Root directory containing experiments
        pattern: Glob pattern to find log directories

    Returns:
        Dictionary mapping experiment name to log directory
    """
    search_path = os.path.join(root_dir, pattern)
    log_dirs = glob.glob(search_path)

    experiments = {}
    for log_dir in log_dirs:
        # Extract experiment name from path
        exp_name = os.path.basename(os.path.dirname(log_dir))
        experiments[exp_name] = log_dir

    return experiments


def aggregate_statistics(experiments: dict) -> pd.DataFrame:
    """
    Aggregate statistics across multiple experiments.

    Args:
        experiments: Dictionary mapping experiment name to log directory

    Returns:
        DataFrame with aggregated statistics
    """
    results = {}

    for exp_name, log_dir in experiments.items():
        print(f"Processing {exp_name}...", end=" ")

        try:
            df = load_selection_history(log_dir)

            # Compute statistics
            selection_counts = df['cam_uid'].value_counts()

            # Gini coefficient
            sorted_counts = sorted(selection_counts.values)
            n = len(sorted_counts)
            cumsum = sum(sorted_counts)
            gini = (2 * sum((i + 1) * count for i, count in enumerate(sorted_counts))) / (n * cumsum) - (n + 1) / n

            # Entropy
            import numpy as np
            probs = selection_counts.values / selection_counts.sum()
            entropy = -sum(probs * np.log2(probs + 1e-10))
            max_entropy = np.log2(len(selection_counts))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            # Coverage
            n_cameras = df['cam_uid'].nunique()
            all_camera_ids = set(range(n_cameras))  # Assuming cameras are 0, 1, 2, ...
            selected_cameras = set(df['cam_uid'].unique())
            coverage = len(selected_cameras) / len(all_camera_ids) if all_camera_ids else 1.0

            results[exp_name] = {
                'Total Selections': len(df),
                'Unique Cameras': n_cameras,
                'Mean Count': selection_counts.mean(),
                'Std Count': selection_counts.std(),
                'Min Count': selection_counts.min(),
                'Max Count': selection_counts.max(),
                'Gini Coefficient': gini,
                'Normalized Entropy': normalized_entropy,
                'Coverage': coverage,
            }

            print("✓")

        except Exception as e:
            print(f"✗ ({e})")
            continue

    df = pd.DataFrame(results).T
    return df


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    root_dir = sys.argv[1]

    if not os.path.exists(root_dir):
        print(f"Error: Root directory not found: {root_dir}")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"Comparing selectors in: {root_dir}")
    print(f"{'=' * 70}\n")

    # Find all experiments
    experiments = find_experiment_logs(root_dir)

    if not experiments:
        print(f"No experiments found in {root_dir}")
        print(f"Expected structure: {root_dir}/<experiment_name>/logs/selection_history.jsonl")
        sys.exit(1)

    print(f"Found {len(experiments)} experiments:")
    for name in experiments.keys():
        print(f"  - {name}")
    print()

    # Create output directory
    output_dir = os.path.join(root_dir, 'comparison')
    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate comparison plot
    print("Generating comparison plot...")
    compare_strategies(experiments, os.path.join(output_dir, 'strategy_comparison.png'))

    # 2. Aggregate statistics
    print("\nAggregating statistics...")
    stats_df = aggregate_statistics(experiments)

    # 3. Save results
    stats_csv = os.path.join(output_dir, 'aggregated_stats.csv')
    stats_df.to_csv(stats_csv)
    print(f"✓ Saved to {stats_csv}")

    # 4. Display summary
    print(f"\n{'=' * 70}")
    print("Aggregated Statistics:")
    print(f"{'=' * 70}\n")
    print(stats_df.to_string())

    # 5. Highlight best performers
    print(f"\n{'=' * 70}")
    print("Best Performers:")
    print(f"{'=' * 70}\n")

    if 'Normalized Entropy' in stats_df.columns:
        best_uniform = stats_df['Normalized Entropy'].idxmax()
        print(f"  Most uniform: {best_uniform} (entropy: {stats_df.loc[best_uniform, 'Normalized Entropy']:.3f})")

    if 'Gini Coefficient' in stats_df.columns:
        most_equal = stats_df['Gini Coefficient'].idxmin()
        print(f"  Most equal: {most_equal} (Gini: {stats_df.loc[most_equal, 'Gini Coefficient']:.3f})")

    if 'Coverage' in stats_df.columns:
        best_coverage = stats_df['Coverage'].idxmax()
        print(f"  Best coverage: {best_coverage} (coverage: {stats_df.loc[best_coverage, 'Coverage']:.1%})")

    print(f"\n{'=' * 70}")
    print(f"✓ Comparison complete! Results saved to {output_dir}")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()
