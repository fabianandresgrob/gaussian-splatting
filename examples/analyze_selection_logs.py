#!/usr/bin/env python3
"""
Example script for analyzing view selection logs.

Usage:
    python analyze_selection_logs.py <log_dir> [output_dir]

Example:
    python analyze_selection_logs.py output/garden_loss_based/logs
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from view_selection.analysis import (
    load_selection_history,
    plot_selection_frequency,
    plot_selection_over_time,
    plot_probability_evolution,
    export_summary_stats,
    create_analysis_report,
    compare_strategies,
)


def analyze_single_experiment(log_dir: str, output_dir: str = None):
    """Analyze a single experiment's selection logs."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(log_dir), 'analysis')

    print(f"\n{'=' * 70}")
    print(f"Analyzing: {log_dir}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 70}\n")

    # Create full analysis report
    create_analysis_report(log_dir, output_dir)

    print(f"\n{'=' * 70}")
    print(f"✓ Analysis complete!")
    print(f"{'=' * 70}\n")


def compare_multiple_experiments(experiment_dirs: dict, output_path: str):
    """Compare multiple experiments side-by-side."""
    print(f"\n{'=' * 70}")
    print(f"Comparing {len(experiment_dirs)} experiments")
    print(f"{'=' * 70}\n")

    for name, log_dir in experiment_dirs.items():
        print(f"  - {name}: {log_dir}")

    compare_strategies(experiment_dirs, output_path)

    print(f"\n✓ Comparison saved to {output_path}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    log_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(log_dir):
        print(f"Error: Log directory not found: {log_dir}")
        sys.exit(1)

    analyze_single_experiment(log_dir, output_dir)


if __name__ == '__main__':
    main()
