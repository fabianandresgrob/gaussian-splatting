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
import os
import subprocess
import sys
from datetime import datetime
from typing import List, Optional

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
IMBALANCE_RATIOS = [30, 60, 90]

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
                "--sensor", sensor,
                "--keep_count", str(count),
                "--strategy", "every_n",
            ]
            
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
                "--sensor", sensor,
                "--subset_fraction", str(ratio / 100.0),
                # Let the script pick a random focus view from train set
            ]
            
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
    ]
    
    if disable_densification:
        cmd.append("--disable_densification")
    
    if densification_multiplier != 1.0:
        cmd.extend(["--densification_multiplier", str(densification_multiplier)])
    
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
    
    # Step 1: Create sparse scenes
    print("\n--- Creating sparse scenes ---")
    sparse_scenes = create_sparse_scenes(
        data_root=data_root,
        scenes=scenes,
        sparse_counts=sparse_counts,
        dry_run=dry_run,
        sensor=sensor,
    )
    
    if create_scenes_only:
        print("\n[DONE] Sparse scenes created. Skipping ablation runs.")
        return
    
    # Step 2: Run ablation on original scenes (baseline)
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
    
    # Step 1: Create imbalanced scenes
    print("\n--- Creating imbalanced scenes ---")
    imbalanced_scenes = create_imbalanced_scenes(
        data_root=data_root,
        scenes=scenes,
        imbalance_ratios=imbalance_ratios,
        dry_run=dry_run,
        sensor=sensor,
    )
    
    if create_scenes_only:
        print("\n[DONE] Imbalanced scenes created. Skipping ablation runs.")
        return
    
    # Step 2: Run ablation on original scenes (baseline)
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
    
    # Optimizer-specific options  
    parser.add_argument("--optimizers", type=str, nargs="+", default=OPTIMIZER_CONFIGS,
                        help=f"Optimizers to test (default: {OPTIMIZER_CONFIGS})")
    
    # Execution options
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be run without executing")
    parser.add_argument("--no_resume", action="store_true",
                        help="Don't skip already completed runs")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
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
    print(f"Data root: {args.data_root}")
    print(f"Output root: {args.output_root}")
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
        )
    
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
        )
    
    print("\n" + "="*80)
    print("ABLATION SUITE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
