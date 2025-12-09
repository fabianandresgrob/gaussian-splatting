# View Selection Analysis Tools

Offline analysis tools for visualizing and comparing camera selection patterns from training logs.

## Quick Start

```python
from view_selection.analysis import quick_analysis

# Analyze a single training run
stats = quick_analysis('/path/to/experiment/logs')
```

This creates:
- `selection_frequency.png`: Bar chart of camera selection counts
- `selection_heatmap.png`: Heatmap showing selections over time
- `probability_evolution.png`: How probabilities changed during training
- `summary_stats.json`: Statistical summary (Gini, entropy, coverage, etc.)

## Available Functions

### 1. Load Selection History

```python
from view_selection.analysis import load_selection_history

df = load_selection_history('/path/to/logs')
# Returns pandas DataFrame with columns:
# - iteration: Training iteration
# - cam_uid: Camera unique ID
# - cam_name: Camera image name
# - probability: Selection probability at that iteration
# - selection_count: Cumulative selection count
```

### 2. Plot Selection Frequency

Shows which cameras were selected most/least often.

```python
from view_selection.analysis import plot_selection_frequency

plot_selection_frequency(
    df,
    output_path='figures/selection_freq.png',
    top_k=20  # Show only top 20 cameras (optional)
)
```

### 3. Plot Selection Over Time (Heatmap)

Visualize selection patterns across training. Great for seeing phase transitions in `ScheduledHybridSelector`.

```python
from view_selection.analysis import plot_selection_over_time

plot_selection_over_time(
    df,
    output_path='figures/selection_heatmap.png',
    bin_size=1000  # Bin iterations (default: 1000)
)
```

### 4. Plot Probability Evolution

Track how selection probabilities changed over time.

```python
from view_selection.analysis import plot_probability_evolution

plot_probability_evolution(
    df,
    output_path='figures/prob_evolution.png',
    top_k=10,  # Show top 10 cameras by variance
    window_size=100  # Smoothing window
)
```

### 5. Plot Camera Positions (3D)

Visualize camera positions colored by selection frequency.

```python
from view_selection.analysis import plot_camera_positions_colored

plot_camera_positions_colored(
    cameras=scene.getTrainCameras(),
    selection_counts=df['cam_uid'].value_counts().to_dict(),
    output_path='figures/camera_positions.png'
)
```

### 6. Compare Multiple Strategies

Side-by-side comparison of different selection strategies.

```python
from view_selection.analysis import compare_strategies

strategies = {
    'Random': '/path/to/random/logs',
    'Clustering': '/path/to/clustering/logs',
    'Loss-Based': '/path/to/loss_based/logs',
    'Hybrid': '/path/to/hybrid/logs',
}

compare_strategies(strategies, 'figures/strategy_comparison.png')
```

### 7. Export Summary Statistics

Compute quantitative metrics for selection patterns.

```python
from view_selection.analysis import export_summary_stats

stats = export_summary_stats(df, 'results/stats.json')

# Returns dict with:
# - gini_coefficient: Inequality measure (0=equal, 1=unequal)
# - entropy: Randomness measure
# - normalized_entropy: Entropy normalized to [0, 1]
# - coverage: % of cameras selected at least once
# - neglected_cameras: List of rarely-selected camera IDs
# - selection_counts: Mean, std, min, max, median
```

**Interpreting metrics**:
- **Gini coefficient**: 0 = perfectly uniform, 1 = all selections on one camera
- **Normalized entropy**: 1 = uniform distribution, 0 = concentrated
- **Coverage**: Should be 100% (1.0) for most strategies
- **Neglected cameras**: Cameras with < 1% of expected selections

### 8. Visualize Weight Schedules

For `ScheduledHybridSelector`, visualize how weights change over time.

```python
from view_selection.analysis import plot_weight_schedule

config = {
    'selectors': ['clustering', 'loss_based'],
    'weights_start': [0.8, 0.2],
    'weights_end': [0.2, 0.8],
    'schedule_type': 'linear',
    'max_iterations': 30000,
}

plot_weight_schedule(
    config=config,
    max_iterations=30000,
    output_path='figures/weight_schedule.png'
)
```

## Integration with Training

### During Training

Selectors automatically write logs to `log_dir/selection_history.jsonl`:

```python
from view_selection import build_selector

selector = build_selector(
    'loss_based',
    log_dir='output/experiment_1/logs',  # Enable logging
    verbose=True
)
```

### After Training

Analyze the logs:

```python
from view_selection.analysis import create_analysis_report

create_analysis_report(
    log_dir='output/experiment_1/logs',
    output_dir='output/experiment_1/analysis',
    cameras=scene.getTrainCameras()  # Optional: for 3D plot
)
```

## Colab/Notebook Integration

```python
# At the end of your training notebook
import os
from view_selection.analysis import *

# Experiment settings
STRATEGIES = ['random', 'clustering', 'loss_based', 'scheduled_hybrid']
SCENE = 'garden'
OUTPUT_ROOT = '/content/drive/MyDrive/gaussian_splatting_results'

# Analyze each strategy
for strategy in STRATEGIES:
    log_dir = f"{OUTPUT_ROOT}/{SCENE}_{strategy}/seed_0/logs"

    if os.path.exists(log_dir):
        print(f"\n[Analyzing] {strategy}")

        df = load_selection_history(log_dir)

        # Individual plots
        plot_selection_frequency(df, f"{OUTPUT_ROOT}/figures/{strategy}_freq.png")
        plot_selection_over_time(df, f"{OUTPUT_ROOT}/figures/{strategy}_heatmap.png")
        plot_probability_evolution(df, f"{OUTPUT_ROOT}/figures/{strategy}_prob.png")

        # Statistics
        stats = export_summary_stats(df, f"{OUTPUT_ROOT}/stats/{strategy}_stats.json")

# Compare all strategies
strategy_logs = {
    s: f"{OUTPUT_ROOT}/{SCENE}_{s}/seed_0/logs"
    for s in STRATEGIES
}

compare_strategies(
    strategy_logs,
    f"{OUTPUT_ROOT}/figures/strategy_comparison.png"
)

print("\n✓ Analysis complete! Check the figures/ directory.")
```

## Example Analysis Workflow

```python
from view_selection.analysis import *
import matplotlib.pyplot as plt

# 1. Load selection history
df = load_selection_history('output/experiment_1/logs')

print(f"Total selections: {len(df)}")
print(f"Unique cameras: {df['cam_uid'].nunique()}")
print(f"Iteration range: [{df['iteration'].min()}, {df['iteration'].max()}]")

# 2. Basic statistics
selection_counts = df['cam_uid'].value_counts()
print(f"\nSelection counts:")
print(f"  Mean: {selection_counts.mean():.1f}")
print(f"  Std: {selection_counts.std():.1f}")
print(f"  Min: {selection_counts.min()}")
print(f"  Max: {selection_counts.max()}")

# 3. Generate visualizations
os.makedirs('figures', exist_ok=True)

plot_selection_frequency(df, 'figures/freq.png', top_k=20)
plot_selection_over_time(df, 'figures/heatmap.png', bin_size=500)
plot_probability_evolution(df, 'figures/prob_evolution.png', top_k=5)

# 4. Export detailed statistics
stats = export_summary_stats(df, 'results/stats.json')

print(f"\nGini coefficient: {stats['gini_coefficient']:.3f}")
print(f"Normalized entropy: {stats['normalized_entropy']:.3f}")
print(f"Coverage: {stats['coverage']:.1%}")

# 5. Compare with baseline
baseline_df = load_selection_history('output/baseline/logs')

compare_strategies(
    {
        'Experiment': 'output/experiment_1/logs',
        'Baseline': 'output/baseline/logs'
    },
    'figures/comparison.png'
)
```

## Batch Processing Multiple Experiments

```python
import glob
from view_selection.analysis import quick_analysis

# Find all experiment directories
experiment_dirs = glob.glob('output/*/logs')

results = {}

for exp_dir in experiment_dirs:
    exp_name = exp_dir.split('/')[-2]
    print(f"\nAnalyzing {exp_name}...")

    try:
        stats = quick_analysis(exp_dir)
        results[exp_name] = stats
    except Exception as e:
        print(f"  Error: {e}")

# Aggregate results
import pandas as pd

summary_df = pd.DataFrame({
    exp_name: {
        'Gini': stats['gini_coefficient'],
        'Entropy': stats['normalized_entropy'],
        'Coverage': stats['coverage'],
        'Mean_Count': stats['selection_counts']['mean'],
        'Std_Count': stats['selection_counts']['std'],
    }
    for exp_name, stats in results.items()
}).T

print("\n=== Summary Across Experiments ===")
print(summary_df)

# Save aggregated results
summary_df.to_csv('results/aggregate_summary.csv')
```

## Dependencies

```bash
pip install pandas matplotlib numpy
```

All dependencies are already included in the main environment.

## Tips

1. **Use `bin_size` wisely**: For long training runs (>30k iterations), use larger bin sizes (2000-5000) in heatmaps

2. **Check entropy**: High entropy (>0.8) indicates uniform sampling, low entropy (<0.5) indicates focused sampling

3. **Monitor Gini**: Values >0.3 suggest significant selection bias

4. **Compare strategies**: Always compare against random baseline to validate improvements

5. **3D camera plots**: Useful for understanding geometric biases in selection

## Troubleshooting

**Problem**: `FileNotFoundError: selection_history.jsonl`

**Solution**: Ensure `log_dir` was set when creating the selector:
```python
selector = build_selector('loss_based', log_dir='path/to/logs')
```

**Problem**: Empty or corrupted log file

**Solution**: Check that training ran long enough and completed successfully. Logs are written incrementally.

**Problem**: Memory error with large logs

**Solution**: Use `chunksize` parameter with pandas:
```python
def load_large_history(log_file, chunksize=10000):
    chunks = pd.read_json(log_file, lines=True, chunksize=chunksize)
    return pd.concat(chunks, ignore_index=True)
```
