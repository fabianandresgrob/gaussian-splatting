#!/usr/bin/env python3
"""Cleanup utility for ablation output directories.

Designed for the folder layout produced by run_ablation.py:
  <output_root>/<CONFIG_ID>/<SCENE>/seed_<SEED>/

Default goal (safe + matches disk-minimal policy):
- Delete point cloud dumps:   point_cloud/
- Delete PyTorch checkpoints: chkpnt*.pth
- Keep final rendered images: eval/, combine/
- Keep small JSONs/logs: final_results.json, run_metadata.json, metrics_history.json, cameras.json, etc.

Safety:
- Default is DRY RUN (no deletions) unless --apply is passed.

Checkpoint policy:
- If a run is completed (final_results.json exists): delete ALL chkpnt*.pth.
- If a run is incomplete:
    - By default keep the newest checkpoint (preferring *_interrupt/_error if present)
    - Use --delete_all_checkpoints to delete all even for incomplete runs.

Examples:
  # Dry run for an output dir
  python cleanup_outputs.py --output_root /path/to/output

  # Apply deletions
  python cleanup_outputs.py --output_root /path/to/output --apply

  # Also delete local wandb artifacts
  python cleanup_outputs.py --output_root /path/to/output --apply --delete_wandb

  # Aggressive: delete all checkpoints, even for incomplete runs
  python cleanup_outputs.py --output_root /path/to/output --apply --delete_all_checkpoints
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


@dataclass
class CleanupDecision:
    run_dir: str
    completed: bool
    delete_dirs: List[str]
    delete_files: List[str]
    keep_files: List[str]


def _is_run_dir(path: str) -> bool:
    """Heuristic: a run dir is a directory named seed_*."""
    base = os.path.basename(path)
    return os.path.isdir(path) and base.startswith("seed_")


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += _file_size(os.path.join(root, name))
    return total


def _human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(n)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024.0
    return f"{n}B"


def _find_latest_checkpoint(run_dir: str) -> Optional[str]:
    candidates = glob.glob(os.path.join(run_dir, "chkpnt*.pth"))
    if not candidates:
        return None

    def _score(path: str) -> Tuple[int, int]:
        base = os.path.basename(path)
        it = 0
        try:
            rest = base[len("chkpnt"):]
            rest = rest.split(".pth")[0]
            num = ""
            for ch in rest:
                if ch.isdigit():
                    num += ch
                else:
                    break
            it = int(num) if num else 0
        except Exception:
            it = 0

        if "_interrupt" in base:
            suffix_prio = 2
        elif "_error" in base:
            suffix_prio = 1
        else:
            suffix_prio = 0
        return (it, suffix_prio)

    return max(candidates, key=_score)


def _iter_run_dirs(output_root: str) -> Iterable[str]:
    """Yield run directories under output_root.

    We treat any directory named seed_* as a run directory.
    """
    output_root = os.path.abspath(output_root)
    for root, dirs, _files in os.walk(output_root):
        # Avoid walking huge point clouds unnecessarily
        dirs[:] = [d for d in dirs if d != "point_cloud"]
        for d in dirs:
            if d.startswith("seed_"):
                yield os.path.join(root, d)


def _collect_run_dirs(path: str, limit: Optional[int] = None) -> List[str]:
    """Collect run directories from a path.

    Supported inputs:
    - A single run directory: .../seed_0
    - A folder containing seed_* subfolders: .../<SCENE>/
    - A full output root (recursively searched): .../output/
    """
    path = os.path.abspath(path)

    if _is_run_dir(path):
        return [path]

    # If this directory directly contains seed_* subfolders, use them
    try:
        children = [
            os.path.join(path, d)
            for d in os.listdir(path)
            if d.startswith("seed_")
        ]
    except OSError:
        children = []

    direct_runs = [c for c in children if _is_run_dir(c)]
    if direct_runs:
        direct_runs.sort()
        return direct_runs[:limit] if limit is not None else direct_runs

    # Otherwise recursively search for seed_* folders
    run_dirs: List[str] = []
    for rd in _iter_run_dirs(path):
        if _is_run_dir(rd):
            run_dirs.append(rd)
        if limit is not None and len(run_dirs) >= limit:
            break
    run_dirs.sort()
    return run_dirs


def _build_decision(
    run_dir: str,
    *,
    delete_wandb: bool,
    delete_all_checkpoints: bool,
    keep_latest_checkpoint_for_incomplete: bool,
) -> CleanupDecision:
    completed = os.path.exists(os.path.join(run_dir, "final_results.json"))

    delete_dirs: List[str] = []
    delete_files: List[str] = []
    keep_files: List[str] = []

    pc_dir = os.path.join(run_dir, "point_cloud")
    if os.path.isdir(pc_dir):
        delete_dirs.append(pc_dir)

    if delete_wandb:
        wb_dir = os.path.join(run_dir, "wandb")
        if os.path.isdir(wb_dir):
            delete_dirs.append(wb_dir)

    checkpoints = sorted(glob.glob(os.path.join(run_dir, "chkpnt*.pth")))
    if checkpoints:
        if completed:
            delete_files.extend(checkpoints)
        else:
            if delete_all_checkpoints:
                delete_files.extend(checkpoints)
            elif keep_latest_checkpoint_for_incomplete:
                latest = _find_latest_checkpoint(run_dir)
                for ckpt in checkpoints:
                    if latest and os.path.abspath(ckpt) == os.path.abspath(latest):
                        keep_files.append(ckpt)
                    else:
                        delete_files.append(ckpt)
            else:
                keep_files.extend(checkpoints)

    return CleanupDecision(
        run_dir=run_dir,
        completed=completed,
        delete_dirs=delete_dirs,
        delete_files=delete_files,
        keep_files=keep_files,
    )


def _apply_decision(decision: CleanupDecision, *, apply: bool) -> Tuple[int, int]:
    """Return (bytes_freed, items_deleted)."""
    bytes_freed = 0
    items_deleted = 0

    for d in decision.delete_dirs:
        if os.path.isdir(d):
            bytes_freed += _dir_size(d)
            items_deleted += 1
            if apply:
                shutil.rmtree(d, ignore_errors=True)

    for f in decision.delete_files:
        if os.path.isfile(f):
            bytes_freed += _file_size(f)
            items_deleted += 1
            if apply:
                try:
                    os.remove(f)
                except OSError:
                    pass

    return bytes_freed, items_deleted


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cleanup large artifacts from ablation output directories")
    p.add_argument(
        "--path",
        default=None,
        help="Path to clean: can be output root, scene folder, or a single seed_* run folder",
    )
    p.add_argument(
        "--output_root",
        default=None,
        help="Backward-compatible alias for --path (root output directory)",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files (default is dry run)",
    )
    p.add_argument(
        "--delete_wandb",
        action="store_true",
        help="Also delete local wandb/ directories inside each run",
    )
    p.add_argument(
        "--delete_all_checkpoints",
        action="store_true",
        help="Delete all chkpnt*.pth even for incomplete runs",
    )
    p.add_argument(
        "--keep_latest_checkpoint_for_incomplete",
        action="store_true",
        default=True,
        help="Keep the newest checkpoint for incomplete runs (default: True)",
    )
    p.add_argument(
        "--no_keep_latest_checkpoint_for_incomplete",
        action="store_true",
        help="Do not keep any checkpoint automatically for incomplete runs (only applies if not using --delete_all_checkpoints)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N run directories (debugging)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    target_path = args.path or args.output_root
    if not target_path:
        raise SystemExit("Error: provide --path (or --output_root)")

    output_root = os.path.abspath(target_path)
    apply = bool(args.apply)
    keep_latest = bool(args.keep_latest_checkpoint_for_incomplete) and (not args.no_keep_latest_checkpoint_for_incomplete)

    run_dirs = _collect_run_dirs(output_root, limit=args.limit)

    mode = "APPLY" if apply else "DRY RUN"
    print(f"[{mode}] Scanning: {output_root}")
    print(f"Found {len(run_dirs)} run directories")

    total_bytes = 0
    total_items = 0
    completed_count = 0
    incomplete_count = 0

    for run_dir in run_dirs:
        decision = _build_decision(
            run_dir,
            delete_wandb=bool(args.delete_wandb),
            delete_all_checkpoints=bool(args.delete_all_checkpoints),
            keep_latest_checkpoint_for_incomplete=keep_latest,
        )

        if decision.completed:
            completed_count += 1
        else:
            incomplete_count += 1

        bytes_freed, items_deleted = _apply_decision(decision, apply=apply)
        total_bytes += bytes_freed
        total_items += items_deleted

        if bytes_freed > 0:
            status = "completed" if decision.completed else "incomplete"
            print(f"- {run_dir} ({status}): would free {_human_bytes(bytes_freed)}")
            if decision.keep_files:
                kept = ", ".join(os.path.basename(p) for p in decision.keep_files)
                print(f"  keep: {kept}")

    print("\nSummary")
    print(f"- Runs: {len(run_dirs)} (completed={completed_count}, incomplete={incomplete_count})")
    print(f"- Items to delete: {total_items}")
    print(f"- Space to free: {_human_bytes(total_bytes)}")
    print(f"- Mode: {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


