"""
Offline analysis tools for view selection logs.

This module provides functions to analyze and visualize camera selection patterns
from training logs. All selectors write JSONL logs (selection_history.jsonl) which
can be analyzed post-training.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D


def load_selection_history(log_dir: str) -> pd.DataFrame:
    """
    Load selection history from JSONL log file.

    Args:
        log_dir: Directory containing selection_history.jsonl

    Returns:
        DataFrame with columns: iteration, cam_uid, cam_name, probability, selection_count

    Raises:
        FileNotFoundError: If log file doesn't exist
    """
    log_file = os.path.join(log_dir, "selection_history.jsonl")

    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Selection log not found: {log_file}")

    # Read JSONL file
    records = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"Empty log file: {log_file}")

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Ensure required columns exist
    required_cols = ['iteration', 'cam_uid', 'probability']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in log file")

    return df


def plot_selection_frequency(
    df: pd.DataFrame,
    output_path: str,
    top_k: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Create bar plot of camera selection frequencies.

    Args:
        df: DataFrame from load_selection_history()
        output_path: Path to save figure (PNG)
        top_k: Only show top K most selected cameras (None = show all)
        figsize: Figure size in inches
    """
    # Count selections per camera
    selection_counts = df['cam_uid'].value_counts().sort_values(ascending=False)

    if top_k is not None:
        selection_counts = selection_counts.head(top_k)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(range(len(selection_counts)), selection_counts.values, color='steelblue')

    # Highlight most and least selected
    if len(bars) > 0:
        bars[0].set_color('darkred')  # Most selected
        if len(bars) > 1:
            bars[-1].set_color('lightblue')  # Least selected

    ax.set_xlabel('Camera UID')
    ax.set_ylabel('Selection Count')
    title = f'Camera Selection Frequency'
    if top_k:
        title += f' (Top {top_k})'
    ax.set_title(title)
    ax.set_xticks(range(len(selection_counts)))
    ax.set_xticklabels(selection_counts.index, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add statistics
    mean_count = selection_counts.mean()
    std_count = selection_counts.std()
    ax.axhline(mean_count, color='red', linestyle='--', alpha=0.5, label=f'Mean: {mean_count:.1f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Selection frequency plot saved to {output_path}")


def plot_selection_over_time(
    df: pd.DataFrame,
    output_path: str,
    bin_size: int = 1000,
    figsize: Tuple[int, int] = (14, 8)
) -> None:
    """
    Create heatmap showing which cameras were selected over time.

    Args:
        df: DataFrame from load_selection_history()
        output_path: Path to save figure (PNG)
        bin_size: Number of iterations per bin
        figsize: Figure size in inches
    """
    # Create iteration bins
    df['iteration_bin'] = (df['iteration'] // bin_size) * bin_size

    # Create pivot table: bins x cameras
    pivot = df.pivot_table(
        values='iteration',
        index='cam_uid',
        columns='iteration_bin',
        aggfunc='count',
        fill_value=0
    )

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    # Set labels
    ax.set_xlabel('Iteration (binned)')
    ax.set_ylabel('Camera UID')
    ax.set_title(f'Camera Selection Over Time (bin size: {bin_size})')

    # Set ticks
    n_bins = len(pivot.columns)
    tick_spacing = max(1, n_bins // 10)
    tick_indices = range(0, n_bins, tick_spacing)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([pivot.columns[i] for i in tick_indices], rotation=45, ha='right')

    # Y-axis: show subset of camera IDs
    n_cams = len(pivot.index)
    if n_cams > 20:
        # Show every Nth camera
        tick_spacing_y = max(1, n_cams // 20)
        tick_indices_y = range(0, n_cams, tick_spacing_y)
        ax.set_yticks(tick_indices_y)
        ax.set_yticklabels([pivot.index[i] for i in tick_indices_y])
    else:
        ax.set_yticks(range(n_cams))
        ax.set_yticklabels(pivot.index)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Selection Count per Bin', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Selection over time heatmap saved to {output_path}")


def plot_probability_evolution(
    df: pd.DataFrame,
    output_path: str,
    top_k: int = 10,
    window_size: int = 100,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot how selection probabilities evolve over time.

    Args:
        df: DataFrame from load_selection_history()
        output_path: Path to save figure (PNG)
        top_k: Number of cameras to show (highest variance)
        window_size: Rolling average window size
        figsize: Figure size in inches
    """
    # Find cameras with highest probability variance
    prob_variance = df.groupby('cam_uid')['probability'].var().sort_values(ascending=False)
    top_cameras = prob_variance.head(top_k).index

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each camera's probability over time
    colors = cm.tab10(np.linspace(0, 1, top_k))

    for i, cam_uid in enumerate(top_cameras):
        cam_data = df[df['cam_uid'] == cam_uid].sort_values('iteration')

        # Apply rolling average for smoothness
        if len(cam_data) > window_size:
            probs_smooth = cam_data['probability'].rolling(window=window_size, center=True).mean()
        else:
            probs_smooth = cam_data['probability']

        ax.plot(
            cam_data['iteration'],
            probs_smooth,
            label=f'Cam {cam_uid}',
            color=colors[i],
            linewidth=2,
            alpha=0.7
        )

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Selection Probability')
    ax.set_title(f'Probability Evolution (Top {top_k} cameras by variance)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, None])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Probability evolution plot saved to {output_path}")


def plot_camera_positions_colored(
    cameras: List,
    selection_counts: Dict[int, int],
    output_path: str,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Create 3D scatter plot of camera positions colored by selection count.

    Args:
        cameras: List of Camera objects with camera_center attribute
        selection_counts: Dictionary mapping cam.uid to selection count
        output_path: Path to save figure (PNG)
        figsize: Figure size in inches
    """
    # Extract positions and counts
    positions = []
    counts = []

    for cam in cameras:
        pos = cam.camera_center.cpu().numpy() if hasattr(cam.camera_center, 'cpu') else cam.camera_center
        positions.append(pos)
        counts.append(selection_counts.get(cam.uid, 0))

    positions = np.array(positions)
    counts = np.array(counts)

    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Normalize counts for coloring
    if counts.max() > 0:
        norm = Normalize(vmin=counts.min(), vmax=counts.max())
        colors = cm.RdYlBu_r(norm(counts))
    else:
        colors = 'gray'

    # Scatter plot
    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=counts,
        cmap='RdYlBu_r',
        s=100,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Positions (colored by selection count)')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Selection Count', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Camera positions plot saved to {output_path}")


def compare_strategies(
    log_dirs: Dict[str, str],
    output_path: str,
    figsize: Optional[Tuple[int, int]] = None
) -> None:
    """
    Compare selection patterns of multiple strategies.

    Args:
        log_dirs: Dictionary mapping strategy name to log directory
        output_path: Path to save figure (PNG)
        figsize: Figure size in inches (auto-sized if None)
    """
    n_strategies = len(log_dirs)

    if figsize is None:
        figsize = (12, 4 * n_strategies)

    fig, axes = plt.subplots(n_strategies, 1, figsize=figsize)

    if n_strategies == 1:
        axes = [axes]

    for ax, (strategy_name, log_dir) in zip(axes, log_dirs.items()):
        try:
            df = load_selection_history(log_dir)
            selection_counts = df['cam_uid'].value_counts().sort_index()

            # Bar plot for this strategy
            ax.bar(selection_counts.index, selection_counts.values, color='steelblue', alpha=0.7)
            ax.set_xlabel('Camera UID')
            ax.set_ylabel('Selection Count')
            ax.set_title(f'Strategy: {strategy_name}')
            ax.grid(axis='y', alpha=0.3)

            # Add statistics
            mean_count = selection_counts.mean()
            ax.axhline(mean_count, color='red', linestyle='--', alpha=0.5,
                      label=f'Mean: {mean_count:.1f}')
            ax.legend()

        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading {strategy_name}:\n{str(e)}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Strategy: {strategy_name} (ERROR)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Strategy comparison plot saved to {output_path}")


def export_summary_stats(df: pd.DataFrame, output_path: str) -> Dict:
    """
    Compute and export summary statistics of selection patterns.

    Args:
        df: DataFrame from load_selection_history()
        output_path: Path to save JSON file

    Returns:
        Dictionary with statistics
    """
    selection_counts = df['cam_uid'].value_counts()
    total_selections = len(df)
    n_cameras = df['cam_uid'].nunique()

    # Compute statistics
    stats = {
        'total_selections': int(total_selections),
        'n_cameras': int(n_cameras),
        'n_cameras_selected': int(len(selection_counts)),
        'coverage': float(len(selection_counts) / n_cameras) if n_cameras > 0 else 0.0,

        'selection_counts': {
            'mean': float(selection_counts.mean()),
            'std': float(selection_counts.std()),
            'min': int(selection_counts.min()),
            'max': int(selection_counts.max()),
            'median': float(selection_counts.median()),
        },

        'probabilities': {
            'mean': float(df['probability'].mean()),
            'std': float(df['probability'].std()),
            'min': float(df['probability'].min()),
            'max': float(df['probability'].max()),
        }
    }

    # Gini coefficient (measure of inequality)
    sorted_counts = np.sort(selection_counts.values)
    n = len(sorted_counts)
    cumsum = np.cumsum(sorted_counts)
    gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_counts)) / (n * cumsum[-1]) - (n + 1) / n
    stats['gini_coefficient'] = float(gini)

    # Selection entropy (measure of uniformity)
    probs = selection_counts.values / selection_counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(len(selection_counts))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    stats['entropy'] = float(entropy)
    stats['normalized_entropy'] = float(normalized_entropy)

    # Neglected cameras (< 1% of expected selections)
    expected_count = total_selections / n_cameras
    threshold = 0.01 * expected_count
    neglected_cameras = [int(uid) for uid, count in selection_counts.items() if count < threshold]
    stats['neglected_cameras'] = neglected_cameras
    stats['n_neglected_cameras'] = len(neglected_cameras)
    stats['neglected_percentage'] = float(len(neglected_cameras) / n_cameras * 100) if n_cameras > 0 else 0.0

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Summary statistics saved to {output_path}")
    print(f"  Gini coefficient: {stats['gini_coefficient']:.3f} (0=equal, 1=unequal)")
    print(f"  Normalized entropy: {stats['normalized_entropy']:.3f} (0=concentrated, 1=uniform)")
    print(f"  Coverage: {stats['coverage']:.1%}")
    print(f"  Neglected cameras: {stats['n_neglected_cameras']}/{n_cameras}")

    return stats


def plot_weight_schedule(
    config: dict,
    max_iterations: int,
    output_path: str,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Visualize weight schedule for ScheduledHybridSelector.

    Args:
        config: Configuration dict for ScheduledHybridSelector
        max_iterations: Maximum number of iterations to plot
        output_path: Path to save figure (PNG)
        figsize: Figure size in inches
    """
    selector_names = config['selectors']
    schedule_type = config['schedule_type']

    # Compute weights directly without creating full selector
    iterations = np.linspace(0, max_iterations, 500)
    weights_over_time = {name: [] for name in selector_names}

    for iteration in iterations:
        iter_int = int(iteration)

        # Compute weights based on schedule type
        if schedule_type == 'linear':
            t = min(iter_int / config['max_iterations'], 1.0)
            weights = [
                start + (end - start) * t
                for start, end in zip(config['weights_start'], config['weights_end'])
            ]

        elif schedule_type == 'cosine':
            t = min(iter_int / config['max_iterations'], 1.0)
            cosine_factor = 0.5 * (1.0 + np.cos(np.pi * t))
            weights = [
                end + (start - end) * cosine_factor
                for start, end in zip(config['weights_start'], config['weights_end'])
            ]

        elif schedule_type == 'step':
            # Find applicable milestone
            milestones = config['milestones']
            current_weights = milestones[0][1]  # Default to first
            for milestone_iter, milestone_weights in milestones:
                if iter_int >= milestone_iter:
                    current_weights = milestone_weights
            weights = current_weights

        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        # Store weights
        for i, name in enumerate(selector_names):
            weights_over_time[name].append(weights[i])

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    colors = cm.tab10(np.linspace(0, 1, len(selector_names)))

    for i, name in enumerate(selector_names):
        ax.plot(iterations, weights_over_time[name], label=name, color=colors[i], linewidth=2)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Weight')
    ax.set_title(f'Weight Schedule: {schedule_type}')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])

    # Add phase markers for step schedule
    if schedule_type == 'step' and 'milestones' in config:
        for milestone_iter, _ in config['milestones']:
            if milestone_iter > 0:
                ax.axvline(milestone_iter, color='red', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Weight schedule plot saved to {output_path}")


def create_analysis_report(
    log_dir: str,
    output_dir: str,
    cameras: Optional[List] = None
) -> None:
    """
    Create a complete analysis report with all visualizations.

    Args:
        log_dir: Directory containing selection_history.jsonl
        output_dir: Directory to save all outputs
        cameras: Optional list of Camera objects for 3D position plot
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[Analysis] Creating report for {log_dir}")
    print("=" * 60)

    try:
        # Load data
        df = load_selection_history(log_dir)
        print(f"✓ Loaded {len(df)} selections")

        # Generate all plots
        plot_selection_frequency(df, os.path.join(output_dir, 'selection_frequency.png'))
        plot_selection_over_time(df, os.path.join(output_dir, 'selection_heatmap.png'))
        plot_probability_evolution(df, os.path.join(output_dir, 'probability_evolution.png'))

        # Export statistics
        stats = export_summary_stats(df, os.path.join(output_dir, 'summary_stats.json'))

        # Camera positions (if provided)
        if cameras is not None:
            selection_counts = df['cam_uid'].value_counts().to_dict()
            plot_camera_positions_colored(
                cameras,
                selection_counts,
                os.path.join(output_dir, 'camera_positions.png')
            )

        print("=" * 60)
        print(f"✓ Analysis report complete! Saved to {output_dir}")

    except Exception as e:
        print(f"✗ Error creating analysis report: {e}")
        raise


# Convenience function for Colab/Notebook usage
def quick_analysis(log_dir: str, output_dir: str = None) -> Dict:
    """
    Quick analysis with default settings.

    Args:
        log_dir: Directory containing selection logs
        output_dir: Output directory (defaults to log_dir/analysis)

    Returns:
        Dictionary with summary statistics
    """
    if output_dir is None:
        output_dir = os.path.join(log_dir, 'analysis')

    create_analysis_report(log_dir, output_dir)

    # Return stats
    stats_path = os.path.join(output_dir, 'summary_stats.json')
    with open(stats_path, 'r') as f:
        return json.load(f)
