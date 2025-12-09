# ScheduledHybridSelector

Combines multiple view selectors with time-dependent weights, enabling adaptive training strategies that evolve over time.

## Concept

Instead of using a single selector for the entire training, this approach transitions between different strategies based on a schedule. This allows you to:

- **Start with diversity** (clustering) → **transition to difficulty** (loss-based)
- **Begin with geometric heuristics** → **shift to Gaussian-aware optimization**
- **Multi-phase training** with different priorities at each stage

## Standard Presets

### 1. Explore-Then-Exploit (`explore_then_exploit`)

**Strategy**: Start with diverse exploration, transition to focusing on difficult views.

```python
from view_selection import build_selector

config = {'preset': 'explore_then_exploit'}
selector = build_selector('scheduled_hybrid', config=config, verbose=True)
```

**Details**:
- Selectors: `clustering` (80%) → `loss_based` (80%)
- Schedule: Linear interpolation over 30k iterations
- Use case: General-purpose adaptive training

**Weight progression**:
```
Iteration 0:     clustering: 0.8, loss_based: 0.2
Iteration 15000: clustering: 0.5, loss_based: 0.5
Iteration 30000: clustering: 0.2, loss_based: 0.8
```

### 2. Geometric-to-Gaussian (`geometric_to_gaussian`)

**Strategy**: Start with camera pose heuristics, transition to Gaussian-aware selection.

```python
config = {'preset': 'geometric_to_gaussian'}
selector = build_selector('scheduled_hybrid', config=config, verbose=True)
```

**Details**:
- Selectors: `fixed_prob` (90%) → `gaussian_aware` (70%)
- Schedule: Cosine annealing (smooth transition)
- Use case: Leveraging scene geometry early, then Gaussian distribution

**Weight progression** (cosine):
```
Iteration 0:     fixed_prob: 0.9, gaussian_aware: 0.1
Iteration 15000: fixed_prob: 0.6, gaussian_aware: 0.4
Iteration 30000: fixed_prob: 0.3, gaussian_aware: 0.7
```

### 3. Three-Phase Training (`three_phase_training`)

**Strategy**: Explicit three-phase training with distinct priorities.

```python
config = {'preset': 'three_phase_training'}
selector = build_selector('scheduled_hybrid', config=config, verbose=True)
```

**Details**:
- Selectors: `clustering`, `fixed_prob`, `loss_based`
- Schedule: Step schedule (piecewise constant)
- Use case: Structured training with clear phases

**Phases**:
```
Phase 1 (0-10k):    clustering: 0.6, fixed_prob: 0.3, loss_based: 0.1  [Diversity]
Phase 2 (10k-20k):  clustering: 0.3, fixed_prob: 0.4, loss_based: 0.3  [Balance]
Phase 3 (20k-30k):  clustering: 0.1, fixed_prob: 0.2, loss_based: 0.7  [Refinement]
```

## Custom Configurations

### Linear Schedule

```python
config = {
    'selectors': ['clustering', 'loss_based', 'gaussian_aware'],
    'weights_start': [0.5, 0.3, 0.2],   # Initial weights
    'weights_end': [0.1, 0.3, 0.6],     # Final weights
    'schedule_type': 'linear',
    'max_iterations': 30000,
    'temperature': 1.0,
    # Optional: sub-selector configs
    'clustering_config': {'n_clusters': 10},
    'loss_based_config': {'ema_decay': 0.95},
}

selector = build_selector('scheduled_hybrid', config=config)
```

### Cosine Schedule

```python
config = {
    'selectors': ['fixed_prob', 'epoch_based'],
    'weights_start': [0.8, 0.2],
    'weights_end': [0.3, 0.7],
    'schedule_type': 'cosine',  # Smooth transition
    'max_iterations': 30000,
}

selector = build_selector('scheduled_hybrid', config=config)
```

### Step Schedule (Multi-Phase)

```python
config = {
    'selectors': ['random', 'clustering', 'loss_based'],
    'schedule_type': 'step',
    'milestones': [
        [0,     [0.5, 0.4, 0.1]],   # Phase 1
        [5000,  [0.3, 0.5, 0.2]],   # Phase 2
        [15000, [0.1, 0.4, 0.5]],   # Phase 3
        [25000, [0.0, 0.2, 0.8]],   # Phase 4
    ],
}

selector = build_selector('scheduled_hybrid', config=config)
```

## Usage in Training Loop

```python
from view_selection import build_selector

# Initialize with preset
config = {'preset': 'explore_then_exploit'}
selector = build_selector('scheduled_hybrid', config=config, verbose=True)
selector.initialize(scene.getTrainCameras())

# Training loop
for iteration in range(30000):
    # Select camera (weights automatically adjusted based on iteration)
    viewpoint_cam = selector.select_view(gaussians, iteration)

    # Render and compute loss
    render_pkg = render(viewpoint_cam, gaussians, ...)
    loss = compute_loss(render_pkg, gt_image)

    # Backprop and optimize
    loss.backward()
    optimizer.step()

    # IMPORTANT: Forward updates to sub-selectors
    selector.update_loss(viewpoint_cam, loss.item())  # For loss_based sub-selector

    # If using gaussian_aware sub-selector
    if iteration % 100 == 0:
        selector.update_coverage_counts(gaussians)

    # Log statistics periodically
    if iteration % 5000 == 0:
        selector.log_statistics(iteration)
```

## How It Works

### Weight Computation

**Linear Schedule**:
```python
t = iteration / max_iterations
weight(t) = start_weight + (end_weight - start_weight) * t
```

**Cosine Schedule** (smooth transition):
```python
t = iteration / max_iterations
cosine_factor = 0.5 * (1 + cos(π * t))
weight(t) = end_weight + (start_weight - end_weight) * cosine_factor
```

**Step Schedule**:
```python
# Piecewise constant weights
# Find milestone where iteration >= milestone_iteration
weight = milestone_weights
```

### Probability Combination

For each iteration:
1. Compute current weights from schedule
2. Get probabilities from each sub-selector
3. Combine with weighted average: `P_combined = Σ(weight_i * P_i)`
4. Renormalize to ensure sum = 1.0

### Update Forwarding

The hybrid selector automatically forwards updates to sub-selectors:
- `update_loss(camera, loss)` → forwarded to `LossBasedSelector`
- `update_coverage_counts(gaussians)` → forwarded to `GaussianAwareSelector`

This ensures sub-selectors maintain their internal state correctly.

## When to Use Each Preset

| Preset | Best For | Training Stage Focus |
|--------|----------|---------------------|
| `explore_then_exploit` | General scenes | Diversity → Difficulty |
| `geometric_to_gaussian` | Complex geometry | Pose → Gaussian density |
| `three_phase_training` | Large scenes | Structured progression |

## Monitoring Progress

### Verbose Logging

Enable verbose mode to see weight evolution:

```python
selector = build_selector('scheduled_hybrid', config=config, verbose=True)
```

Output every 1000 iterations:
```
[ScheduledHybridSelector] Iteration 5000:
  Current weights: ['0.600', '0.400']
  Selectors: ['clustering', 'loss_based']
```

### Statistics

```python
# Get current statistics
stats = selector.get_hybrid_statistics(iteration)

# Print detailed statistics
selector.log_statistics(iteration)
```

## Advantages

✅ **Adaptive training**: Strategy evolves with scene complexity
✅ **Combines strengths**: Leverage multiple selector types
✅ **Smooth transitions**: Cosine schedule prevents abrupt changes
✅ **Multi-phase control**: Step schedule for explicit phases
✅ **Easy presets**: Standard configs for common scenarios

## Best Practices

1. **Start diverse, end focused**: Begin with exploration (clustering), end with refinement (loss-based)

2. **Match schedule to training length**:
   - Short training (<10k): Use linear or step
   - Long training (>20k): Use cosine for smooth transition

3. **Monitor sub-selectors**: Check that loss updates and coverage tracking work

4. **Customize weights**: Adjust presets based on your scene characteristics

5. **Log regularly**: Use `verbose=True` and `log_statistics()` to track evolution

## Example: Custom Adaptive Strategy

```python
# Early: Focus on diversity and geometry
# Mid: Balance all strategies
# Late: Focus on Gaussian coverage and loss

config = {
    'selectors': ['clustering', 'fixed_prob', 'gaussian_aware', 'loss_based'],
    'schedule_type': 'step',
    'milestones': [
        [0,     [0.4, 0.3, 0.2, 0.1]],  # Early: diversity
        [10000, [0.2, 0.2, 0.3, 0.3]],  # Mid: balanced
        [20000, [0.1, 0.1, 0.4, 0.4]],  # Late: Gaussian+loss
    ],
    'gaussian_aware_config': {
        'mode': 'coverage_gap',
        'update_frequency': 500,
    },
    'loss_based_config': {
        'ema_decay': 0.95,
        'min_samples_before_bias': 10,
    },
}

selector = build_selector('scheduled_hybrid', config=config, verbose=True)
```

This creates a sophisticated 4-selector strategy with explicit control over each phase!
