# GaussianAwareSelector

Adaptive view selection based on the current 3D Gaussian distribution.

## Concept

Instead of relying only on camera poses or training loss, this selector adapts to the actual Gaussian scene representation. It prioritizes cameras based on which Gaussians they can see, enabling targeted training of under-represented or difficult regions.

## Modes

### 1. Inverse Density Mode (`inverse_density`)

Prioritizes cameras that see **fewer Gaussians** (sparse regions).

**Rationale**: Regions with fewer Gaussians might be under-represented or more challenging to reconstruct. By prioritizing cameras viewing these sparse areas, we can improve coverage of difficult regions.

**Use case**: Early training stages, scenes with varying density.

### 2. Coverage Gap Mode (`coverage_gap`)

Prioritizes cameras seeing **under-trained Gaussians** (based on training counts).

**Rationale**: Some Gaussians might be visible from many cameras but rarely trained. This mode tracks how often each Gaussian has been trained and prioritizes cameras viewing under-trained Gaussians.

**Use case**: Later training stages, ensuring uniform Gaussian training coverage.

## Configuration

```python
from view_selection import build_selector

# Inverse density mode (default)
config = {
    'mode': 'inverse_density',        # or 'coverage_gap'
    'update_frequency': 500,           # Update Gaussian stats every N iterations
    'temperature': 1.0,                # Softmax temperature for probabilities
    'frustum_margin': 1.2,             # Frustum check margin (1.0 = exact, >1.0 = more inclusive)
    'near_plane': 0.01,                # Near clipping plane
    'far_plane': 100.0,                # Far clipping plane
}

selector = build_selector('gaussian_aware', config=config, verbose=True, seed=42)
```

## Usage in Training Loop

```python
# Initialize
selector.initialize(scene.getTrainCameras())

# Training loop
for iteration in range(iterations):
    # Select camera (Gaussian stats computed automatically every update_frequency)
    viewpoint_cam = selector.select_view(gaussians, iteration)

    # Render and compute loss
    render_pkg = render(viewpoint_cam, gaussians, ...)
    loss = compute_loss(render_pkg, gt_image)

    # Backprop and optimize
    loss.backward()
    optimizer.step()

    # IMPORTANT: Update coverage tracking for coverage_gap mode
    # Option 1: Update every iteration (if single camera)
    if selector.mode == 'coverage_gap':
        selector.update_coverage_counts(gaussians, viewpoint_cam)

    # Option 2: Batch update every N iterations (recommended for performance)
    if selector.mode == 'coverage_gap' and iteration % 100 == 0:
        selector.update_coverage_counts(gaussians)  # Processes pending queue
```

## How It Works

### Frustum Culling

For each camera, we determine which Gaussians are visible using frustum check:

1. **Transform** Gaussian positions to camera space using `world_view_transform`
2. **Depth check**: Keep only Gaussians with `z > near_plane` and `z < far_plane`
3. **Projection**: Project to normalized device coordinates (NDC)
4. **Bounds check**: Keep only Gaussians within image bounds (with margin)
   - Uses actual camera FOV (`FoVx`, `FoVy`) if available for accurate frustum
   - Falls back to fixed bounds if FOV not available

This is implemented as a **vectorized batch operation** using PyTorch for efficiency.

### Coverage Tracking (coverage_gap mode)

Coverage tracking uses a **lazy batch processing** approach for performance:

1. Selected cameras are added to a **pending queue** in `log_selection()`
2. `update_coverage_counts()` processes the queue periodically (recommended: every 100 iterations)
3. Coverage counts automatically reset when Gaussian count changes (densification/pruning)

This minimizes overhead while maintaining accurate coverage statistics.

### Probability Computation

**Inverse Density:**
```
score(camera) = 1 / (num_visible_gaussians + 1)
probability = softmax(scores / temperature)
```

**Coverage Gap:**
```
score(camera) = 1 / (mean_coverage_of_visible_gaussians + 1)
probability = softmax(scores / temperature)
```

## Performance Considerations

Frustum checking all Gaussians for all cameras is **expensive**. This selector uses several optimizations:

1. **Caching**: Gaussian visibility is only recomputed every `update_frequency` iterations (default: 500)
2. **Batch operations**: All computations use vectorized PyTorch operations
3. **Lazy updates**: Coverage tracking only updates when explicitly called

**Recommended settings:**
- `update_frequency`: 500-1000 for large scenes (>100k Gaussians)
- `update_frequency`: 100-500 for smaller scenes (<50k Gaussians)

## Statistics

Track Gaussian visibility and coverage statistics:

```python
# Get statistics
stats = selector.get_gaussian_statistics()

# Print statistics
selector.log_statistics(iteration)
```

**Example output:**
```
[GaussianAwareSelector] Statistics at iteration 5000:
  Mode: inverse_density
  Selection stats:
    Total selections: 5000
    Unique cameras: 248
  Gaussian visibility:
    Range: [1234, 5678]
    Mean ± std: 3456.7 ± 892.3
  Gaussian coverage:
    Range: [0.0, 45.2]
    Mean ± std: 12.3 ± 8.7
```

## When to Use

**Good for:**
- ✅ Scenes with varying Gaussian density
- ✅ Complex scenes with under-represented regions
- ✅ Later training stages (refinement)
- ✅ Ensuring uniform Gaussian coverage

**Not ideal for:**
- ❌ Very early training (few Gaussians exist yet)
- ❌ Extremely large scenes (>500k Gaussians) without sufficient GPU memory
- ❌ When training speed is critical (overhead from frustum checks)

## Comparison with Other Selectors

| Selector | Uses Gaussians | Adapts During Training | Overhead |
|----------|----------------|------------------------|----------|
| Random | ❌ | ❌ | None |
| FixedProbability | ❌ | ❌ | Low |
| EpochBased | ❌ | ✅ | Low |
| Clustering | ❌ | Limited | Low |
| LossBased | Indirectly | ✅ | Low |
| **GaussianAware** | **✅** | **✅** | **Medium** |

## Advanced: Combining with Other Strategies

You can combine GaussianAwareSelector with other strategies by switching selectors during training:

```python
# Early training: Use clustering for diversity
selector_early = build_selector('clustering', config={'n_clusters': 10})

# Late training: Use Gaussian-aware for refinement
selector_late = build_selector('gaussian_aware', config={'mode': 'coverage_gap'})

# Switch at iteration 15000
if iteration < 15000:
    cam = selector_early.select_view(gaussians, iteration)
else:
    cam = selector_late.select_view(gaussians, iteration)
```
