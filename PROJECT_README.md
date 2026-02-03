# Project README (View Selection for 3D Gaussian Splatting)

This document summarizes the project-specific additions to the original 3D Gaussian Splatting codebase. It focuses on the view-selection strategies, how to run them, and how to reproduce the ablation studies.

## Overview

We add configurable **view selection strategies** that replace uniform random sampling during training. The goal is to study whether more informed sampling improves training efficiency or final quality. The project includes:

- A modular selector interface (`view_selection/`)
- Multiple strategies (geometric, loss-based, DINO features, clustering, etc.)
- A training entry point that supports selecting and logging strategies (`train_gsplat.py`)
- A single-run ablation runner (`run_ablation.py`)
- Multi-axis ablation suites (`run_ablation_suite.py`)
- Utilities in `tools/` to generate specialized datasets and analyze results

## View Selection Strategies

All strategies implement a common interface in `view_selection/selector.py` and can be selected via `--view_selection_strategy` and `--view_selection_config` in `train_gsplat.py`.

Available strategies (key behavior):

- `stack` (baseline): Original 3DGS behavior. Shuffle all cameras into a stack and pop without replacement per epoch.
- `uniform_random` (baseline): Uniform sampling with replacement.
- `sequential` (baseline): Deterministic round-robin over cameras.
- `geometric`: Prioritizes pose diversity using camera position and optional orientation. Dynamic mode biases away from recently selected views.
- `loss_based`: Tracks the most recent rendering loss per camera and biases toward higher-loss views (after a minimum sampling warm-up).
- `dino`: Uses DINO embeddings to prefer semantically diverse views (distance to recent selections or centroid).
- `clustering`: Clusters cameras by pose (KMeans or DBSCAN) and assigns probability inversely proportional to cluster size.
- `deterministic_max_loss`: Renders all cameras per iteration and selects the max-loss camera (deterministic, very expensive).

Selector names are registered in `view_selection/__init__.py` and can be listed at runtime.

## How to Run Training With a Strategy

Typical training command (example uses DINO):

```bash
python train_gsplat.py \
  --source_path /path/to/scene/dslr \
  --model_path /path/to/output/run1 \
  --view_selection_strategy dino \
  --view_selection_config '{"embeddings_path":"auto","temperature":0.3,"diversity_mode":"distance_to_selected"}' \
  --iterations 30000
```

Notes:
- `view_selection_config` is a JSON string.
- If `embeddings_path` is set to `"auto"`, DINO embeddings are inferred from the dataset layout.
- Use `--no_view_selection_verbose` to reduce selector logging noise.

## Running the Ablations

### Single ablation driver

`run_ablation.py` orchestrates experiments across multiple scenes, seeds, and strategies.

Config IDs (from `run_ablation.py`):

- `B1`: Baseline stack (original 3DGS epoch-based shuffle)
- `B2`: Baseline uniform random sampling
- `S1`: Geometric diversity (pose-based, dynamic distance to recent selections)
- `S2`: Loss-based sampling (bias toward high-loss views after warm-up)
- `S3`: DINO feature diversity (distance to recent selections)
- `CL`: Clustering (DBSCAN pose clustering, inverse cluster-size weighting)
- `SEQ`: Sequential (deterministic round-robin)
- `DML`: Deterministic max-loss (renders all views each step; expensive)

Example:

```bash
python run_ablation.py \
  --data_root /path/to/scenes \
  --output_root /path/to/outputs \
  --scenes 0c5385e84b 5371eff4f9 \
  --configs B1 B2 S1 S2 S3 CL \
  --seeds 0 1 2
```

Helpful options:

- `--dry_run`: show planned runs
- `--list_configs`: list available config IDs and parameters
- `--no_resume`: disable resume behavior
- `--no_view_selection_verbose`: quieter selector logs
- `--full_disk`: save checkpoints/point clouds (default is minimal disk usage)

### Ablation suites (multi-axis experiments)

`run_ablation_suite.py` runs multi-axis suites (sparse views, densification settings, optimizer variants, imbalanced data).

Example:

```bash
python run_ablation_suite.py \
  --data_root /path/to/scenes \
  --output_root /path/to/outputs \
  --suite sparse
```

Suites:

- `sparse`: create reduced-view datasets (e.g., 50/25/10) and run ablations
- `densification`: vary densification thresholds or disable it
- `optimizer`: compare Adam vs SGD variants
- `imbalance`: introduce duplicate views and test selector robustness

Each suite internally calls `run_ablation.py` with consistent config sets.

## Tools (Utilities)

The `tools/` folder includes utilities used by the ablation workflows and analysis:

- `extract_dinov3_features.py`: Precompute DINOv3 embeddings for DINO-based selection.
- `sparsify_scene_views.py`: Create reduced-view datasets (used in sparse ablations).
- `imbalance_scene_views.py`: Create datasets with duplicated views (used in imbalance ablations).
- `generate_test_split.py`: Generate custom train/test splits if needed.
- `cleanup_outputs.py`: Remove unnecessary artifacts from output directories.
- `analyze_ablation_selection.py`: Analyze selection logs and compute selection distribution statistics.

These tools can be run standalone, but are also integrated into `run_ablation_suite.py`.

## Quick Reference

- Training entry point: `train_gsplat.py`
- Single ablation runner: `run_ablation.py`
- Ablation suites: `run_ablation_suite.py`
- Strategy implementations: `view_selection/`
- Utilities: `tools/`

If you need a specific strategy config or a minimal reproducible run, check the configs defined in `run_ablation.py`.
