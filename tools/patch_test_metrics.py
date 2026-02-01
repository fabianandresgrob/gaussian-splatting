#!/usr/bin/env python3
"""Patch test metrics from fixed runs into original flawed runs.

This script updates test metrics from re-run experiments (with correct test.txt)
into the original flawed runs, while PRESERVING all train metrics.

What happens:
  - final_results.json: MERGED - keeps train_* keys from original,
                        replaces test metrics (mean_psnr, etc.) with fixed values
  - metrics_history.json: MERGED - keeps train metrics from original,
                          replaces test metrics with fixed values
  - split_summary.json: copied from source (just metadata)

Examples
--------
Patch sparse ablation results:

  python tools/patch_test_metrics.py \
    --source_dir output_mip360_ablation_suites_fixed/sparse_baseline_original \
    --dest_dir output_mip360_ablation_suites/sparse_baseline_original \
    --dry_run

  # If looks good, run without --dry_run:
  python tools/patch_test_metrics.py \
    --source_dir output_mip360_ablation_suites_fixed/sparse_baseline_original \
    --dest_dir output_mip360_ablation_suites/sparse_baseline_original

Patch multiple directories:

  python tools/patch_test_metrics.py \
    --source_dir output_mip360_ablation_suites_fixed/sparse_50_views \
    --dest_dir output_mip360_ablation_suites/sparse_50_views
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple, Optional


def discover_runs(ablation_dir: Path) -> List[Tuple[str, str, int, Path]]:
    """Discover all runs in an ablation directory.
    
    Returns:
        List of (config, scene, seed, run_dir) tuples
    """
    runs = []
    
    if not ablation_dir.exists():
        return runs
    
    # Expected structure: ablation_dir/CONFIG/SCENE/seed_N/
    for config_dir in ablation_dir.iterdir():
        if not config_dir.is_dir():
            continue
        if config_dir.name in ('logs', 'wandb'):
            continue
        if config_dir.name.endswith('.json'):
            continue
            
        config = config_dir.name
        
        for scene_dir in config_dir.iterdir():
            if not scene_dir.is_dir():
                continue
            
            scene = scene_dir.name
            
            for seed_dir in scene_dir.iterdir():
                if not seed_dir.is_dir():
                    continue
                if not seed_dir.name.startswith('seed_'):
                    continue
                
                try:
                    seed = int(seed_dir.name.split('_')[1])
                except (IndexError, ValueError):
                    continue
                
                runs.append((config, scene, seed, seed_dir))
    
    return runs


def merge_metrics_history(source_file: Path, dest_file: Path) -> list:
    """Merge metrics history: keep train from dest, take test from source.
    
    Returns:
        Merged metrics history list
    """
    with open(source_file, 'r') as f:
        source_history = json.load(f)
    
    with open(dest_file, 'r') as f:
        dest_history = json.load(f)
    
    # Index source by iteration for fast lookup
    source_by_iter = {entry['iteration']: entry for entry in source_history}
    
    # Merge: keep train from dest, take test from source
    merged = []
    for dest_entry in dest_history:
        iteration = dest_entry['iteration']
        
        merged_entry = {
            'iteration': iteration,
            'elapsed_time': dest_entry.get('elapsed_time'),  # Keep original timing
        }
        
        # Keep train metrics from destination (original)
        if 'train' in dest_entry:
            merged_entry['train'] = dest_entry['train']
        
        # Take test metrics from source (fixed)
        if iteration in source_by_iter and 'test' in source_by_iter[iteration]:
            merged_entry['test'] = source_by_iter[iteration]['test']
        elif 'test' in dest_entry:
            # Fallback to original if source doesn't have this iteration
            merged_entry['test'] = dest_entry['test']
        
        merged.append(merged_entry)
    
    return merged


def merge_final_results(source_file: Path, dest_file: Path) -> dict:
    """Merge final results: keep train_* from dest, take test metrics from source.
    
    Keys like mean_psnr, mean_ssim, mean_lpips are test metrics.
    Keys starting with train_ are train metrics.
    
    Returns:
        Merged final results dict
    """
    with open(source_file, 'r') as f:
        source_data = json.load(f)
    
    with open(dest_file, 'r') as f:
        dest_data = json.load(f)
    
    # Start with source data (the fixed test metrics)
    merged = dict(source_data)
    
    # Add back any train_* keys from destination (original)
    for key, value in dest_data.items():
        if key.startswith('train'):
            merged[key] = value
    
    return merged


def patch_run(
    source_run_dir: Path,
    dest_run_dir: Path,
    dry_run: bool = False,
    verbose: bool = True,
) -> bool:
    """Patch test metrics from source to destination run.
    
    - final_results.json: merged (keep train_* from dest, take test from source)
    - metrics_history.json: merged (keep train from dest, test from source)
    - split_summary.json: copied directly (just metadata about splits)
    
    Returns:
        True if successful, False otherwise
    """
    
    success = True
    
    # 1. Merge final_results.json (keep train_*, replace test metrics)
    source_final = source_run_dir / 'final_results.json'
    dest_final = dest_run_dir / 'final_results.json'
    
    if not source_final.exists():
        if verbose:
            print(f"    [SKIP] final_results.json not found in source")
    elif not dest_final.exists():
        if verbose:
            print(f"    [SKIP] final_results.json not found in destination")
    else:
        if dry_run:
            if verbose:
                print(f"    [DRY-RUN] Would merge final_results.json (keep train_*, replace test)")
        else:
            try:
                # Create backup
                backup_file = dest_final.with_suffix(dest_final.suffix + '.bak')
                if not backup_file.exists():
                    shutil.copy2(dest_final, backup_file)
                
                # Merge and write
                merged = merge_final_results(source_final, dest_final)
                with open(dest_final, 'w') as f:
                    json.dump(merged, f, indent=4)
                
                if verbose:
                    print(f"    [OK] Merged final_results.json (kept train_*, replaced test)")
            except Exception as e:
                print(f"    [ERROR] Failed to merge final_results.json: {e}")
                success = False
    
    # 2. Copy split_summary.json directly (just metadata)
    source_split = source_run_dir / 'split_summary.json'
    dest_split = dest_run_dir / 'split_summary.json'
    
    if not source_split.exists():
        if verbose:
            print(f"    [SKIP] split_summary.json not found in source")
    elif not dest_split.exists():
        if verbose:
            print(f"    [SKIP] split_summary.json not found in destination")
    else:
        if dry_run:
            if verbose:
                print(f"    [DRY-RUN] Would copy split_summary.json")
        else:
            try:
                backup_file = dest_split.with_suffix(dest_split.suffix + '.bak')
                if not backup_file.exists():
                    shutil.copy2(dest_split, backup_file)
                
                shutil.copy2(source_split, dest_split)
                
                if verbose:
                    print(f"    [OK] Copied split_summary.json")
            except Exception as e:
                print(f"    [ERROR] Failed to copy split_summary.json: {e}")
                success = False
    
    # 3. Merge metrics_history.json (keep train, replace test)
    source_history = source_run_dir / 'metrics_history.json'
    dest_history = dest_run_dir / 'metrics_history.json'
    
    if not source_history.exists():
        if verbose:
            print(f"    [SKIP] metrics_history.json not found in source")
    elif not dest_history.exists():
        if verbose:
            print(f"    [SKIP] metrics_history.json not found in destination")
    else:
        if dry_run:
            if verbose:
                print(f"    [DRY-RUN] Would merge metrics_history.json (keep train, replace test)")
        else:
            try:
                # Create backup
                backup_file = dest_history.with_suffix(dest_history.suffix + '.bak')
                if not backup_file.exists():
                    shutil.copy2(dest_history, backup_file)
                
                # Merge and write
                merged = merge_metrics_history(source_history, dest_history)
                with open(dest_history, 'w') as f:
                    json.dump(merged, f, indent=4)
                
                if verbose:
                    print(f"    [OK] Merged metrics_history.json (kept train, replaced test)")
            except Exception as e:
                print(f"    [ERROR] Failed to merge metrics_history.json: {e}")
                success = False
    
    return success


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch test metrics from fixed runs into original flawed runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to FIXED ablation output directory (source of correct metrics)",
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        required=True,
        help="Path to ORIGINAL/FLAWED ablation output directory (to be patched)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show errors and summary",
    )
    
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    dest_dir = Path(args.dest_dir)
    
    # Validate directories
    if not source_dir.exists():
        print(f"[ERROR] Source directory not found: {source_dir}")
        return 1
    
    if not dest_dir.exists():
        print(f"[ERROR] Destination directory not found: {dest_dir}")
        return 1
    
    print(f"\n{'=' * 70}")
    print(f"PATCH TEST METRICS")
    print(f"{'=' * 70}")
    print(f"Source (fixed):      {source_dir}")
    print(f"Destination (flawed): {dest_dir}")
    print(f"Dry run:             {args.dry_run}")
    print(f"{'=' * 70}")
    
    # Discover runs in source
    source_runs = discover_runs(source_dir)
    print(f"\nFound {len(source_runs)} runs in source directory")
    
    # Discover runs in destination
    dest_runs = discover_runs(dest_dir)
    print(f"Found {len(dest_runs)} runs in destination directory")
    
    # Build destination index
    dest_index = {}
    for config, scene, seed, run_dir in dest_runs:
        key = (config, scene, seed)
        dest_index[key] = run_dir
    
    # Patch matching runs
    patched = 0
    skipped = 0
    failed = 0
    
    print(f"\n{'-' * 70}")
    
    for config, scene, seed, source_run_dir in sorted(source_runs):
        key = (config, scene, seed)
        
        if key not in dest_index:
            if not args.quiet:
                print(f"[SKIP] {config}/{scene}/seed_{seed} - not found in destination")
            skipped += 1
            continue
        
        dest_run_dir = dest_index[key]
        
        if not args.quiet:
            print(f"\n[PATCH] {config}/{scene}/seed_{seed}")
        
        success = patch_run(
            source_run_dir,
            dest_run_dir,
            dry_run=args.dry_run,
            verbose=not args.quiet,
        )
        
        if success:
            patched += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"Patched: {patched}")
    print(f"Skipped: {skipped} (not found in destination)")
    print(f"Failed:  {failed}")
    
    if args.dry_run:
        print(f"\n[DRY-RUN] No files were modified. Run without --dry_run to apply changes.")
    else:
        print(f"\n[INFO] Backups created with .bak extension")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
