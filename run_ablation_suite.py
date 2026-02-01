#!/usr/bin/env python3
"""
Ablation Suite Runner - Orchestrates multiple ablation experiments.

This script runs comprehensive ablation studies to understand why intelligent
view selection strategies may not show significant improvements over baselines.

Three main ablation axes:
1. SPARSE: Test on scenes with fewer training views (50, 25, 10 views)
2. DENSIFICATION: Test with different densification settings
3. OPTIMIZER: Test with different optimizers (Adam vs SGD)

Usage:
    # Run all ablation suites
    python run_ablation_suite.py --data_root /path/to/scenes --output_root /path/to/results

    # Run specific suite only
    python run_ablation_suite.py --suite sparse --data_root ... --output_root ...
    python run_ablation_suite.py --suite densification --data_root ... --output_root ...
    python run_ablation_suite.py --suite optimizer --data_root ... --output_root ...

    # Dry run
    python run_ablation_suite.py --dry_run --data_root ... --output_root ...

    # Create sparse scenes only (without running ablations)
    python run_ablation_suite.py --suite sparse --create_scenes_only --data_root ... --output_root ...
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ============================================================================
#                              CONFIGURATION
# ============================================================================

# Scenes to use for ablations (3 diverse scenes)
DEFAULT_SCENES = [
    "0c5385e84b",
    "ab046f8faf",
    "f248c2bcdc",
]

# View selection configs to test
CONFIGS = ["B1", "B2", "S1", "S2", "S3", "CL"]

# Training parameters
ITERATIONS = 7500
SEED = 0

# Test iterations: every 10 up to 1000, every 100 to 5000, every 500 to 7500
TEST_ITERATIONS = (
    list(range(10, 1001, 10)) +      # 10, 20, ..., 1000 (100 values)
    list(range(1100, 5001, 100)) +   # 1100, 1200, ..., 5000 (40 values)
    list(range(5500, 7501, 500))     # 5500, 6000, 6500, 7000, 7500 (5 values)
)

# Sparse scene configurations
SPARSE_COUNTS = [50, 25, 10]

# Imbalance configurations (percentage of train set that becomes duplicates)
IMBALANCE_RATIOS = [80, 90]

# Densification configurations
DENSIFICATION_CONFIGS = {
    "baseline": {"disable": False, "multiplier": 1.0},
    "disabled": {"disable": True, "multiplier": 1.0},
    "threshold_2x": {"disable": False, "multiplier": 2.0},
    "threshold_4x": {"disable": False, "multiplier": 4.0},
}

# Optimizer configurations
OPTIMIZER_CONFIGS = ["default", "sgd", "sgd_no_momentum", "sparse_adam"]


# ============================================================================
#                              HELPER FUNCTIONS
# ============================================================================

def get_sparse_scene_id(scene_id: str, count: int) -> str:
    """Generate sparse scene ID."""
    return f"{scene_id}_sparse_{count}"


def create_sparse_scenes(
    data_root: str,
    scenes: List[str],
    sparse_counts: List[int],
    dry_run: bool = False,
    sensor: str = "dslr",
    format: str = "auto",
    llffhold: int = 8,
    images: Optional[str] = None,
) -> List[str]:
    """Create sparse versions of scenes.
    
    Returns list of created sparse scene IDs.
    """
    created = []
    
    for scene in scenes:
        for count in sparse_counts:
            sparse_id = get_sparse_scene_id(scene, count)
            out_path = os.path.join(data_root, sparse_id)
            
            if os.path.exists(out_path):
                print(f"[SKIP] Sparse scene already exists: {sparse_id}")
                created.append(sparse_id)
                continue
            
            cmd = [
                "python", "tools/sparsify_scene_views.py",
                "--data_root", data_root,
                "--scene_id", scene,
                "--out_scene_id", sparse_id,
                "--format", format,
                "--sensor", sensor,
                "--llffhold", str(llffhold),
                "--keep_count", str(count),
                "--strategy", "every_n",
            ]
            
            if images:
                cmd.extend(["--images", images])
            
            print(f"\n[CREATE] Creating sparse scene: {sparse_id} ({count} views)")
            print(f"  Command: {' '.join(cmd)}")
            
            if not dry_run:
                result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
                if result.returncode == 0:
                    created.append(sparse_id)
                else:
                    print(f"[ERROR] Failed to create sparse scene: {sparse_id}")
            else:
                created.append(sparse_id)
    
    return created


def get_imbalanced_scene_id(scene_id: str, ratio: int) -> str:
    """Generate imbalanced scene ID."""
    return f"{scene_id}_imbalanced_{ratio}"


def create_imbalanced_scenes(
    data_root: str,
    scenes: List[str],
    imbalance_ratios: List[int],
    dry_run: bool = False,
    sensor: str = "dslr",
    format: str = "auto",
    llffhold: int = 8,
    images: Optional[str] = None,
) -> List[str]:
    """Create imbalanced versions of scenes by duplicating views.
    
    Returns list of created imbalanced scene IDs.
    """
    created = []
    
    for scene in scenes:
        for ratio in imbalance_ratios:
            imbalanced_id = get_imbalanced_scene_id(scene, ratio)
            out_path = os.path.join(data_root, imbalanced_id)
            
            if os.path.exists(out_path):
                print(f"[SKIP] Imbalanced scene already exists: {imbalanced_id}")
                created.append(imbalanced_id)
                continue
            
            cmd = [
                "python", "tools/imbalance_scene_views.py",
                "--data_root", data_root,
                "--scene_id", scene,
                "--out_scene_id", imbalanced_id,
                "--format", format,
                "--sensor", sensor,
                "--llffhold", str(llffhold),
                "--subset_fraction", str(ratio / 100.0),
                "--random_focus_count", "1",  # Pick a random focus view from train set
            ]
            
            if images:
                cmd.extend(["--images", images])
            
            print(f"\n[CREATE] Creating imbalanced scene: {imbalanced_id} ({ratio}% duplicates)")
            print(f"  Command: {' '.join(cmd)}")
            
            if not dry_run:
                result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
                if result.returncode == 0:
                    created.append(imbalanced_id)
                else:
                    print(f"[ERROR] Failed to create imbalanced scene: {imbalanced_id}")
            else:
                created.append(imbalanced_id)
    
    return created


def relog_results_to_wandb(
    source_run_dir: str,
    wandb_project: str,
    run_name: str,
    dry_run: bool = False,
) -> bool:
    """Re-log existing results to a new W&B project without retraining.
    
    Args:
        source_run_dir: Directory containing metrics_history.json and final_results.json
        wandb_project: Target W&B project name
        run_name: Name for the W&B run
        dry_run: If True, just print what would happen
        
    Returns:
        True if successful, False otherwise
    """
    metrics_path = os.path.join(source_run_dir, "metrics_history.json")
    final_path = os.path.join(source_run_dir, "final_results.json")
    metadata_path = os.path.join(source_run_dir, "run_metadata.json")
    
    if not os.path.exists(metrics_path):
        print(f"  [WARN] No metrics_history.json found: {source_run_dir}")
        return False
    
    if dry_run:
        print(f"  [DRY RUN] Would re-log {run_name} to project {wandb_project}")
        return True
    
    if not WANDB_AVAILABLE:
        print(f"  [WARN] wandb not available, skipping re-log for {run_name}")
        return False
    
    # Load existing data
    with open(metrics_path, 'r') as f:
        metrics_history = json.load(f)
    
    config = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            config = json.load(f)
    config["relogged"] = True
    config["source_run_dir"] = source_run_dir
    
    # Initialize W&B run
    run = wandb.init(
        project=wandb_project,
        name=run_name,
        config=config,
        reinit=True,
    )
    
    # Log all metrics from history
    for entry in metrics_history:
        iteration = entry.get("iteration", 0)
        log_dict = {k: v for k, v in entry.items() if k != "iteration"}
        wandb.log(log_dict, step=iteration)
    
    # Log final results if available
    if os.path.exists(final_path):
        with open(final_path, 'r') as f:
            final_results = json.load(f)
        wandb.log({"final/" + k: v for k, v in final_results.items()})
    
    wandb.finish()
    print(f"  [RELOGGED] {run_name} -> {wandb_project}")
    return True


def copy_and_relog_baseline(
    source_baseline_dir: str,
    target_output_dir: str,
    wandb_project: str,
    scenes: List[str],
    configs: List[str],
    seed: int,
    dry_run: bool = False,
) -> bool:
    """Copy baseline results and re-log to a new W&B project.
    
    Args:
        source_baseline_dir: Directory containing existing baseline runs (e.g., sparse_baseline_original)
        target_output_dir: Where to copy the results (e.g., densification_baseline)
        wandb_project: Target W&B project name
        scenes: List of scene IDs to look for
        configs: List of config IDs (B1, B2, etc.)
        seed: Random seed used
        dry_run: If True, just print what would happen
        
    Returns:
        True if all runs were successfully copied and re-logged
    """
    if not os.path.exists(source_baseline_dir):
        print(f"  [WARN] Source baseline dir not found: {source_baseline_dir}")
        return False
    
    success = True
    relogged_count = 0
    
    for config_id in configs:
        for scene in scenes:
            # Source path: source_baseline_dir/CONFIG/SCENE/seed_N/
            source_run = os.path.join(source_baseline_dir, config_id, scene, f"seed_{seed}")
            target_run = os.path.join(target_output_dir, config_id, scene, f"seed_{seed}")
            
            # Check if source exists
            final_results = os.path.join(source_run, "final_results.json")
            if not os.path.exists(final_results):
                print(f"  [SKIP] Source not complete: {source_run}")
                continue
            
            # Check if target already exists and is complete
            target_final = os.path.join(target_run, "final_results.json")
            if os.path.exists(target_final):
                print(f"  [SKIP] Target already exists: {target_run}")
                continue
            
            # Extract scene name for run naming (handle sensor suffix)
            scene_name = scene
            if scene_name in ("dslr", "iphone"):
                scene_name = os.path.basename(os.path.dirname(scene))
            
            run_name = f"{scene_name}_{config_id.lower()}_seed{seed}"
            
            if dry_run:
                print(f"  [DRY RUN] Would copy {source_run} -> {target_run}")
                print(f"  [DRY RUN] Would re-log as {run_name} to {wandb_project}")
                relogged_count += 1
                continue
            
            # Copy the run directory
            print(f"  [COPY] {source_run} -> {target_run}")
            os.makedirs(os.path.dirname(target_run), exist_ok=True)
            if os.path.exists(target_run):
                shutil.rmtree(target_run)
            shutil.copytree(source_run, target_run)
            
            # Re-log to W&B
            if relog_results_to_wandb(target_run, wandb_project, run_name, dry_run=False):
                relogged_count += 1
            else:
                success = False
    
    print(f"  [SUMMARY] Re-logged {relogged_count} runs to {wandb_project}")
    return success


def run_ablation(
    data_root: str,
    output_root: str,
    scenes: List[str],
    configs: List[str],
    seed: int,
    iterations: int,
    test_iterations: List[int],
    wandb_project: str,
    ablation_name: str,
    optimizer_type: str = "default",
    disable_densification: bool = False,
    densification_multiplier: float = 1.0,
    dry_run: bool = False,
    extra_args: Optional[List[str]] = None,
    resolution: int = 2,
    images: Optional[str] = None,
    data_device: str = "cpu",
    eval_test_only: bool = False,
) -> int:
    """Run a single ablation experiment.
    
    Returns subprocess return code.
    """
    output_dir = os.path.join(output_root, ablation_name)
    
    cmd = [
        "python", "run_ablation.py",
        "--data_root", data_root,
        "--output_root", output_dir,
        "--scenes", *scenes,
        "--seeds", str(seed),
        "--configs", *configs,
        "--iterations", str(iterations),
        "--test_iterations", *[str(i) for i in test_iterations],
        "--wandb_project", wandb_project,
        "--optimizer_type", optimizer_type,
        "--resolution", str(resolution),
        "--data_device", data_device,
    ]
    
    if images:
        cmd.extend(["--images", images])
    
    if disable_densification:
        cmd.append("--disable_densification")
    
    if densification_multiplier != 1.0:
        cmd.extend(["--densification_multiplier", str(densification_multiplier)])
    
    if eval_test_only:
        cmd.append("--eval_test_only")
    
    if extra_args:
        cmd.extend(extra_args)
    
    if dry_run:
        cmd.append("--dry_run")
    
    print(f"\n{'='*80}")
    print(f"ABLATION: {ablation_name}")
    print(f"{'='*80}")
    print(f"  Scenes: {', '.join(scenes)}")
    print(f"  Configs: {', '.join(configs)}")
    print(f"  Iterations: {iterations}")
    print(f"  Resolution: {resolution} (images: {images or 'default'})")
    print(f"  Data device: {data_device}")
    print(f"  Optimizer: {optimizer_type}")
    print(f"  Densification: {'disabled' if disable_densification else f'enabled (multiplier={densification_multiplier})'}")
    print(f"  Output: {output_dir}")
    print(f"  W&B Project: {wandb_project}")
    print(f"\n  Command: {' '.join(cmd[:20])}...")
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode


# ============================================================================
#                              ABLATION SUITES
# ============================================================================

def run_sparse_ablation(
    data_root: str,
    output_root: str,
    scenes: List[str],
    configs: List[str],
    seed: int,
    iterations: int,
    test_iterations: List[int],
    sparse_counts: List[int],
    dry_run: bool = False,
    create_scenes_only: bool = False,
    sensor: str = "dslr",
    extra_args: Optional[List[str]] = None,
    shared_baseline_dir: Optional[str] = None,
    format: str = "auto",
    llffhold: int = 8,
    resolution: int = 2,
    images: Optional[str] = None,
    data_device: str = "cpu",
    eval_test_only: bool = False,
):
    """Run sparse view ablation suite.
    
    Tests whether view selection strategies provide more benefit
    when training with fewer views (reduced camera overlap).
    """
    print("\n" + "="*80)
    print("SPARSE VIEW ABLATION SUITE")
    print("="*80)
    print(f"Testing sparse counts: {sparse_counts}")
    print(f"Original scenes: {scenes}")
    print(f"Format: {format}")
    
    # Step 1: Create sparse scenes
    print("\n--- Creating sparse scenes ---")
    sparse_scenes = create_sparse_scenes(
        data_root=data_root,
        scenes=scenes,
        sparse_counts=sparse_counts,
        dry_run=dry_run,
        sensor=sensor,
        format=format,
        llffhold=llffhold,
        images=images,
    )
    
    if create_scenes_only:
        print("\n[DONE] Sparse scenes created. Skipping ablation runs.")
        return
    
    # Step 2: Run ablation on original scenes (baseline)
    # This is the canonical baseline - other suites will copy from here
    baseline_output = os.path.join(output_root, "sparse_baseline_original")
    print("\n--- Running baseline (original scenes) ---")
    run_ablation(
        data_root=data_root,
        output_root=output_root,
        scenes=scenes,
        configs=configs,
        seed=seed,
        iterations=iterations,
        test_iterations=test_iterations,
        wandb_project="3dgs-sparse-ablation",
        ablation_name="sparse_baseline_original",
        dry_run=dry_run,
        extra_args=extra_args,
        resolution=resolution,
        images=images,
        data_device=data_device,
        eval_test_only=eval_test_only,
    )
    
    # Step 3: Run ablation on each sparse level
    for count in sparse_counts:
        sparse_scene_ids = [get_sparse_scene_id(s, count) for s in scenes]
        
        # Verify sparse scenes exist
        missing = [s for s in sparse_scene_ids if not os.path.exists(os.path.join(data_root, s))]
        if missing and not dry_run:
            print(f"[WARN] Missing sparse scenes: {missing}")
            continue
        
        print(f"\n--- Running sparse_{count} ablation ---")
        run_ablation(
            data_root=data_root,
            output_root=output_root,
            scenes=sparse_scene_ids,
            configs=configs,
            seed=seed,
            iterations=iterations,
            test_iterations=test_iterations,
            wandb_project="3dgs-sparse-ablation",
            ablation_name=f"sparse_{count}_views",
            dry_run=dry_run,
            extra_args=extra_args,
            resolution=resolution,
            images=images,
            data_device=data_device,
            eval_test_only=eval_test_only,
        )


def run_densification_ablation(
    data_root: str,
    output_root: str,
    scenes: List[str],
    configs: List[str],
    seed: int,
    iterations: int,
    test_iterations: List[int],
    dry_run: bool = False,
    extra_args: Optional[List[str]] = None,
    shared_baseline_dir: Optional[str] = None,
    resolution: int = 2,
    images: Optional[str] = None,
    data_device: str = "cpu",
):
    """Run densification ablation suite.
    
    Tests whether densification "smooths out" the advantage of
    intelligent view selection by adding Gaussians where needed.
    """
    print("\n" + "="*80)
    print("DENSIFICATION ABLATION SUITE")
    print("="*80)
    print(f"Testing densification configs: {list(DENSIFICATION_CONFIGS.keys())}")
    
    for config_name, config in DENSIFICATION_CONFIGS.items():
        target_dir = os.path.join(output_root, f"densification_{config_name}")
        
        # For 'baseline' config, try to reuse shared baseline if available
        if config_name == "baseline" and shared_baseline_dir:
            print(f"\n--- Re-logging densification_baseline from shared baseline ---")
            success = copy_and_relog_baseline(
                source_baseline_dir=shared_baseline_dir,
                target_output_dir=target_dir,
                wandb_project="3dgs-densification-ablation",
                scenes=scenes,
                configs=configs,
                seed=seed,
                dry_run=dry_run,
            )
            if success:
                continue
            print("  [FALLBACK] Shared baseline not fully available, running training...")
        
        print(f"\n--- Running densification_{config_name} ---")
        run_ablation(
            data_root=data_root,
            output_root=output_root,
            scenes=scenes,
            configs=configs,
            seed=seed,
            iterations=iterations,
            test_iterations=test_iterations,
            wandb_project="3dgs-densification-ablation",
            ablation_name=f"densification_{config_name}",
            disable_densification=config["disable"],
            densification_multiplier=config["multiplier"],
            dry_run=dry_run,
            extra_args=extra_args,
            resolution=resolution,
            images=images,
            data_device=data_device,
        )


def run_optimizer_ablation(
    data_root: str,
    output_root: str,
    scenes: List[str],
    configs: List[str],
    seed: int,
    iterations: int,
    test_iterations: List[int],
    optimizer_configs: List[str],
    dry_run: bool = False,
    extra_args: Optional[List[str]] = None,
    shared_baseline_dir: Optional[str] = None,
    resolution: int = 2,
    images: Optional[str] = None,
    data_device: str = "cpu",
):
    """Run optimizer ablation suite.
    
    Tests whether Adam's adaptive learning rates compensate for
    suboptimal view selection, making SGD potentially more sensitive.
    """
    print("\n" + "="*80)
    print("OPTIMIZER ABLATION SUITE")
    print("="*80)
    print(f"Testing optimizers: {optimizer_configs}")
    
    for optimizer in optimizer_configs:
        target_dir = os.path.join(output_root, f"optimizer_{optimizer}")
        
        # For 'default' optimizer (Adam), try to reuse shared baseline if available
        if optimizer == "default" and shared_baseline_dir:
            print(f"\n--- Re-logging optimizer_default from shared baseline ---")
            success = copy_and_relog_baseline(
                source_baseline_dir=shared_baseline_dir,
                target_output_dir=target_dir,
                wandb_project="3dgs-optimizer-ablation",
                scenes=scenes,
                configs=configs,
                seed=seed,
                dry_run=dry_run,
            )
            if success:
                continue
            print("  [FALLBACK] Shared baseline not fully available, running training...")
        
        print(f"\n--- Running optimizer_{optimizer} ---")
        run_ablation(
            data_root=data_root,
            output_root=output_root,
            scenes=scenes,
            configs=configs,
            seed=seed,
            iterations=iterations,
            test_iterations=test_iterations,
            wandb_project="3dgs-optimizer-ablation",
            ablation_name=f"optimizer_{optimizer}",
            optimizer_type=optimizer,
            dry_run=dry_run,
            extra_args=extra_args,
            resolution=resolution,
            images=images,
            data_device=data_device,
        )


def run_imbalance_ablation(
    data_root: str,
    output_root: str,
    scenes: List[str],
    configs: List[str],
    seed: int,
    iterations: int,
    test_iterations: List[int],
    imbalance_ratios: List[int],
    dry_run: bool = False,
    create_scenes_only: bool = False,
    sensor: str = "dslr",
    extra_args: Optional[List[str]] = None,
    shared_baseline_dir: Optional[str] = None,
    format: str = "auto",
    llffhold: int = 8,
    resolution: int = 2,
    images: Optional[str] = None,
    data_device: str = "cpu",
    eval_test_only: bool = False,
):
    """Run imbalance ablation suite.
    
    Tests whether view selection strategies can compensate for
    artificially imbalanced training sets (duplicated views).
    """
    print("\n" + "="*80)
    print("IMBALANCE ABLATION SUITE")
    print("="*80)
    print(f"Testing imbalance ratios: {imbalance_ratios}%")
    print(f"Original scenes: {scenes}")
    print(f"Format: {format}")
    
    # Step 1: Create imbalanced scenes
    print("\n--- Creating imbalanced scenes ---")
    imbalanced_scenes = create_imbalanced_scenes(
        data_root=data_root,
        scenes=scenes,
        imbalance_ratios=imbalance_ratios,
        dry_run=dry_run,
        sensor=sensor,
        format=format,
        llffhold=llffhold,
        images=images,
    )
    
    if create_scenes_only:
        print("\n[DONE] Imbalanced scenes created. Skipping ablation runs.")
        return
    
    # Step 2: Run ablation on original scenes (baseline)
    target_dir = os.path.join(output_root, "imbalance_baseline_original")
    
    # Try to reuse shared baseline if available
    if shared_baseline_dir:
        print("\n--- Re-logging imbalance_baseline_original from shared baseline ---")
        success = copy_and_relog_baseline(
            source_baseline_dir=shared_baseline_dir,
            target_output_dir=target_dir,
            wandb_project="3dgs-imbalance-ablation",
            scenes=scenes,
            configs=configs,
            seed=seed,
            dry_run=dry_run,
        )
        if not success:
            print("  [FALLBACK] Shared baseline not fully available, running training...")
            run_ablation(
                data_root=data_root,
                output_root=output_root,
                scenes=scenes,
                configs=configs,
                seed=seed,
                iterations=iterations,
                test_iterations=test_iterations,
                wandb_project="3dgs-imbalance-ablation",
                ablation_name="imbalance_baseline_original",
                dry_run=dry_run,
                extra_args=extra_args,
                resolution=resolution,
                images=images,
                data_device=data_device,
                eval_test_only=eval_test_only,
            )
    else:
        print("\n--- Running baseline (original scenes) ---")
        run_ablation(
            data_root=data_root,
            output_root=output_root,
            scenes=scenes,
            configs=configs,
            seed=seed,
            iterations=iterations,
            test_iterations=test_iterations,
            wandb_project="3dgs-imbalance-ablation",
            ablation_name="imbalance_baseline_original",
            dry_run=dry_run,
            extra_args=extra_args,
            resolution=resolution,
            images=images,
            data_device=data_device,
            eval_test_only=eval_test_only,
        )
    
    # Step 3: Run ablation on each imbalance level
    for ratio in imbalance_ratios:
        imbalanced_scene_ids = [get_imbalanced_scene_id(s, ratio) for s in scenes]
        
        # Verify imbalanced scenes exist
        missing = [s for s in imbalanced_scene_ids if not os.path.exists(os.path.join(data_root, s))]
        if missing and not dry_run:
            print(f"[WARN] Missing imbalanced scenes: {missing}")
            continue
        
        print(f"\n--- Running imbalanced_{ratio}pct ablation ---")
        run_ablation(
            data_root=data_root,
            output_root=output_root,
            scenes=imbalanced_scene_ids,
            configs=configs,
            seed=seed,
            iterations=iterations,
            test_iterations=test_iterations,
            wandb_project="3dgs-imbalance-ablation",
            ablation_name=f"imbalanced_{ratio}pct",
            dry_run=dry_run,
            extra_args=extra_args,
            resolution=resolution,
            images=images,
            data_device=data_device,
            eval_test_only=eval_test_only,
        )


# ============================================================================
#                              CLI INTERFACE
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ablation study suites for 3DGS view selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all ablation suites
    python run_ablation_suite.py --data_root /path/to/scenes --output_root /path/to/results

    # Run specific suite
    python run_ablation_suite.py --suite sparse --data_root ... --output_root ...

    # Dry run (preview commands)
    python run_ablation_suite.py --dry_run --data_root ... --output_root ...

    # Create sparse scenes only
    python run_ablation_suite.py --suite sparse --create_scenes_only --data_root ... --output_root ...

    # Use custom scenes
    python run_ablation_suite.py --scenes scene1 scene2 scene3 --data_root ... --output_root ...

Ablation Suites:
    sparse         - Test with fewer training views (50, 25, 10)
    imbalance      - Test with artificially imbalanced training sets (30%, 60%, 90% duplicates)
    densification  - Test with different densification settings
    optimizer      - Test with different optimizers (Adam, SGD)
        """
    )
    
    # Required arguments
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing scene data")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root directory for experiment outputs")
    
    # Suite selection
    parser.add_argument("--suite", type=str, nargs="+",
                        choices=["sparse", "imbalance", "densification", "optimizer", "all"],
                        default=["all"],
                        help="Which ablation suite(s) to run (default: all)")
    
    # Scene/config overrides
    parser.add_argument("--scenes", type=str, nargs="+", default=DEFAULT_SCENES,
                        help=f"Scenes to use (default: {DEFAULT_SCENES})")
    parser.add_argument("--configs", type=str, nargs="+", default=CONFIGS,
                        help=f"View selection configs to test (default: {CONFIGS})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    parser.add_argument("--iterations", type=int, default=ITERATIONS,
                        help=f"Training iterations (default: {ITERATIONS})")
    
    # Sparse-specific options
    parser.add_argument("--sparse_counts", type=int, nargs="+", default=SPARSE_COUNTS,
                        help=f"Sparse view counts to test (default: {SPARSE_COUNTS})")
    parser.add_argument("--imbalance_ratios", type=int, nargs="+", default=IMBALANCE_RATIOS,
                        help=f"Imbalance ratios in percent to test (default: {IMBALANCE_RATIOS})")
    parser.add_argument("--create_scenes_only", action="store_true",
                        help="Only create sparse scenes, don't run ablations")
    parser.add_argument("--sensor", type=str, default="dslr",
                        help="Sensor subfolder for ScanNet++ scenes (default: dslr)")
    parser.add_argument("--format", type=str, choices=["auto", "scannetpp", "colmap"], default="auto",
                        help="Dataset format: auto (detect), scannetpp, or colmap/MipNeRF-360 (default: auto)")
    parser.add_argument("--llffhold", type=int, default=8,
                        help="LLFF-hold value for COLMAP format (every Nth for test, default: 8)")
    parser.add_argument("--resolution", type=int, default=2,
                        help="Training resolution: -1 (auto), 1 (full), 2 (half), 4 (quarter). Default: 2")
    parser.add_argument("--images", type=str, default=None,
                        help="Images subfolder for COLMAP datasets (e.g., images_4 for pre-downscaled). "
                             "When set, --resolution is forced to 1.")
    parser.add_argument("--data_device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device for image data: cpu (memory-efficient) or cuda (faster). Default: cpu")
    
    # Optimizer-specific options  
    parser.add_argument("--optimizers", type=str, nargs="+", default=OPTIMIZER_CONFIGS,
                        help=f"Optimizers to test (default: {OPTIMIZER_CONFIGS})")
    
    # Execution options
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be run without executing")
    parser.add_argument("--no_resume", action="store_true",
                        help="Don't skip already completed runs")
    parser.add_argument("--eval_test_only", action="store_true",
                        help="Only evaluate on test set during training (skip train set eval, saves time for large/imbalanced datasets)")
    
    # Shared baseline options
    parser.add_argument("--shared_baseline_dir", type=str, default=None,
                        help="Path to existing baseline results to reuse (e.g., output_root/sparse_baseline_original). "
                             "If set, other suites will copy+re-log these results instead of retraining.")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate image/resolution combinations
    if args.images:
        # When using pre-downscaled images (images_2, images_4, images_8), force resolution=1
        if args.resolution != 1:
            print(f"[NOTE] --images {args.images} specified; forcing --resolution 1 to avoid double-downscaling.")
            args.resolution = 1
        
        # Validate images folder name
        valid_images_folders = ["images", "images_2", "images_4", "images_8"]
        if args.images not in valid_images_folders:
            print(f"[WARN] Non-standard images folder '{args.images}'. Expected one of: {valid_images_folders}")
    
    # Determine which suites to run
    suites = args.suite
    if "all" in suites:
        suites = ["sparse", "imbalance", "densification", "optimizer"]
    
    # Build extra args to pass through
    extra_args = []
    if args.no_resume:
        extra_args.append("--no_resume")
    
    # Build test iterations
    test_iterations = TEST_ITERATIONS
    
    # Determine shared baseline directory
    # By default, use sparse_baseline_original from output_root if it exists
    shared_baseline_dir = args.shared_baseline_dir
    if shared_baseline_dir is None:
        default_baseline = os.path.join(args.output_root, "sparse_baseline_original")
        if os.path.exists(default_baseline):
            shared_baseline_dir = default_baseline
            print(f"Using shared baseline: {shared_baseline_dir}")
    elif shared_baseline_dir:
        print(f"Using shared baseline: {shared_baseline_dir}")
    
    print("="*80)
    print("3DGS VIEW SELECTION ABLATION SUITE")
    print("="*80)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Suites: {suites}")
    print(f"Scenes: {args.scenes}")
    print(f"Configs: {args.configs}")
    print(f"Seed: {args.seed}")
    print(f"Iterations: {args.iterations}")
    print(f"Test iterations: {len(test_iterations)} checkpoints")
    print(f"  First 10: {test_iterations[:10]}")
    print(f"  Last 10: {test_iterations[-10:]}")
    print(f"Resolution: {args.resolution} (images: {args.images or 'default'})")
    print(f"Data device: {args.data_device}")
    print(f"Data root: {args.data_root}")
    print(f"Output root: {args.output_root}")
    print(f"Shared baseline: {shared_baseline_dir or 'None (will train baselines)'}")
    print(f"Dry run: {args.dry_run}")
    print("="*80)
    
    # Run requested suites
    if "sparse" in suites:
        run_sparse_ablation(
            data_root=args.data_root,
            output_root=args.output_root,
            scenes=args.scenes,
            configs=args.configs,
            seed=args.seed,
            iterations=args.iterations,
            test_iterations=test_iterations,
            sparse_counts=args.sparse_counts,
            dry_run=args.dry_run,
            create_scenes_only=args.create_scenes_only,
            sensor=args.sensor,
            extra_args=extra_args,
            shared_baseline_dir=shared_baseline_dir,
            format=args.format,
            llffhold=args.llffhold,
            resolution=args.resolution,
            images=args.images,
            data_device=args.data_device,
            eval_test_only=args.eval_test_only,
        )
        # After sparse runs, update shared_baseline_dir to point to the new baseline
        if shared_baseline_dir is None:
            new_baseline = os.path.join(args.output_root, "sparse_baseline_original")
            if os.path.exists(new_baseline) or args.dry_run:
                shared_baseline_dir = new_baseline
    
    if "imbalance" in suites:
        run_imbalance_ablation(
            data_root=args.data_root,
            output_root=args.output_root,
            scenes=args.scenes,
            configs=args.configs,
            seed=args.seed,
            iterations=args.iterations,
            test_iterations=test_iterations,
            imbalance_ratios=args.imbalance_ratios,
            dry_run=args.dry_run,
            create_scenes_only=args.create_scenes_only,
            sensor=args.sensor,
            extra_args=extra_args,
            shared_baseline_dir=shared_baseline_dir,
            format=args.format,
            llffhold=args.llffhold,
            resolution=args.resolution,
            images=args.images,
            data_device=args.data_device,
            eval_test_only=args.eval_test_only,
        )
    
    if "densification" in suites and not args.create_scenes_only:
        run_densification_ablation(
            data_root=args.data_root,
            output_root=args.output_root,
            scenes=args.scenes,
            configs=args.configs,
            seed=args.seed,
            iterations=args.iterations,
            test_iterations=test_iterations,
            dry_run=args.dry_run,
            extra_args=extra_args,
            shared_baseline_dir=shared_baseline_dir,
            resolution=args.resolution,
            images=args.images,
            data_device=args.data_device,
        )
    
    if "optimizer" in suites and not args.create_scenes_only:
        run_optimizer_ablation(
            data_root=args.data_root,
            output_root=args.output_root,
            scenes=args.scenes,
            configs=args.configs,
            seed=args.seed,
            iterations=args.iterations,
            test_iterations=test_iterations,
            optimizer_configs=args.optimizers,
            dry_run=args.dry_run,
            extra_args=extra_args,
            shared_baseline_dir=shared_baseline_dir,
            resolution=args.resolution,
            images=args.images,
            data_device=args.data_device,
        )
    
    print("\n" + "="*80)
    print("ABLATION SUITE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
