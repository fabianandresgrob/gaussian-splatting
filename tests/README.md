# View Selection Tests

This directory contains unit tests for all view selection strategies.

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_view_selection.py
```

### Run specific test
```bash
pytest tests/test_view_selection.py::test_selector_registry
```

### Run tests for specific selector
```bash
pytest tests/test_view_selection.py -k "random"
```

### Run with verbose output
```bash
pytest -v
```

### Run with coverage
```bash
pytest --cov=view_selection --cov-report=html
```

## Test Coverage

### Test Cases

1. **test_selector_registry**: Verifies all selectors can be instantiated via `build_selector()`
2. **test_probability_sum**: Ensures probabilities always sum to 1.0
3. **test_reproducibility**: Confirms same seed produces identical selection sequences
4. **test_all_cameras_nonzero_prob**: Checks that all cameras have non-zero probability
5. **test_epoch_selector_reset**: Validates EpochBasedSelector resets after N selections
6. **test_clustering_all_assigned**: Ensures all cameras are assigned to clusters (K-Means & DBSCAN)
7. **test_loss_based_selector_updates**: Verifies LossBasedSelector correctly updates EMA losses
8. **test_dbscan_noise_handling**: Tests DBSCAN noise point handling (outlier cluster)
9. **test_selection_statistics**: Validates selection tracking and statistics

### Tested Selectors

- RandomSelector
- FixedProbabilitySelector
- EpochBasedSelector
- ClusteringSelector (K-Means and DBSCAN)
- WithoutReplacementSelector
- LossBasedSelector

## Fixtures

- `mock_cameras_simple`: 10 cameras arranged in a circle
- `mock_cameras_diverse`: 20 cameras with random positions and orientations
- `mock_gaussians`: Mock Gaussian model (placeholder)

## Requirements

- pytest
- numpy
- torch
- scipy
- scikit-learn
- pandas
- matplotlib

Install test dependencies:
```bash
pip install pytest pytest-cov pandas matplotlib
```

## Analysis Tools

The `view_selection/analysis.py` module provides offline analysis of selection logs:

```bash
# Analyze a single experiment
python examples/analyze_selection_logs.py output/garden_loss_based/logs

# Compare multiple strategies
python examples/compare_selectors.py output/
```

See `view_selection/README_ANALYSIS.md` for detailed documentation.
