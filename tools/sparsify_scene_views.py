#!/usr/bin/env python3
"""Create a sparse copy of a ScanNet++-style scene by subsampling training views.

Motivation
----------
To test whether intelligent view selection provides more benefit in sparse-view
settings where camera overlap is reduced. This script creates a copy of a scene
with fewer training images while keeping the test set unchanged.

Sampling strategies:
  - 'every_n': Take every N-th image (deterministic, preserves trajectory structure)
  - 'random': Random sampling of images (uses seed for reproducibility)

The test set is ALWAYS preserved unchanged to ensure fair evaluation.

Examples
--------
Keep every 10th training image (deterministic):

  python tools/sparsify_scene_views.py \
    --data_root /home/fgrob/data/scenes/data \
    --scene_id test_scene \
    --out_scene_id test_scene_sparse_10pct \
    --sensor dslr \
    --keep_fraction 0.10 \
    --strategy every_n

Random sample of 20 training images:

  python tools/sparsify_scene_views.py \
    --data_root /home/fgrob/data/scenes/data \
    --scene_id test_scene \
    --out_scene_id test_scene_sparse_20 \
    --sensor dslr \
    --keep_count 20 \
    --strategy random \
    --seed 42

Notes
-----
- Either --keep_fraction or --keep_count must be specified (not both)
- The test set is always preserved (never subsampled)
- COLMAP images.txt is updated to only contain kept views
- DINO features are updated to only contain kept views
- train_test_lists.json is updated if present
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    import numpy as np
except ImportError:
    np = None


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _copytree_selective(src: Path, dst: Path, exclude_patterns: List[str] = None) -> None:
    """Copy directory tree, optionally excluding certain patterns."""
    exclude_patterns = exclude_patterns or []
    
    def _should_exclude(name: str) -> bool:
        for pattern in exclude_patterns:
            if pattern in name:
                return True
        return False
    
    if dst.exists():
        shutil.rmtree(dst)
    
    shutil.copytree(
        src, dst,
        ignore=lambda d, files: [f for f in files if _should_exclude(f)] if exclude_patterns else []
    )


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def _get_train_test_lists_path(scene_sensor_root: Path) -> Optional[Path]:
    """Return the path to train_test_lists.json (or train_test_list.json) if it exists."""
    candidates = [
        scene_sensor_root / "train_test_lists.json",
        scene_sensor_root / "train_test_list.json",
    ]
    return next((p for p in candidates if p.exists()), None)


def _split_from_train_test_lists(scene_sensor_root: Path) -> Optional[Dict[str, List[str]]]:
    """Load train/test split from train_test_lists.json if it exists."""
    path = _get_train_test_lists_path(scene_sensor_root)
    if path is None:
        return None

    data = _load_json(path)
    if not isinstance(data, dict):
        return None

    train = data.get("train")
    test = data.get("test")
    if not isinstance(train, list) or not isinstance(test, list):
        return None

    # Ensure all are basenames
    train = [os.path.basename(x) for x in train]
    test = [os.path.basename(x) for x in test]

    return {"train": train, "test": test, "has_masks": bool(data.get("has_masks", False))}


def _build_frame_index(transforms: dict) -> Dict[str, dict]:
    """Build an index of all frames by filename."""
    frames = transforms.get("frames", [])
    test_frames = transforms.get("test_frames", [])
    
    index: Dict[str, dict] = {}
    
    for fr in frames:
        if not isinstance(fr, dict):
            continue
        fp = fr.get("file_path")
        if fp:
            index[os.path.basename(fp)] = fr
    
    for fr in test_frames if isinstance(test_frames, list) else []:
        if not isinstance(fr, dict):
            continue
        fp = fr.get("file_path")
        if fp:
            basename = os.path.basename(fp)
            if basename not in index:
                index[basename] = fr
    
    return index


def _select_every_n(items: List[str], keep_fraction: float) -> List[str]:
    """Select every N-th item to achieve approximately keep_fraction."""
    if keep_fraction >= 1.0:
        return items
    if keep_fraction <= 0.0:
        return []
    
    n = max(1, int(round(1.0 / keep_fraction)))
    return items[::n]


def _select_random(items: List[str], keep_count: int, seed: int) -> List[str]:
    """Randomly select keep_count items."""
    if keep_count >= len(items):
        return items
    if keep_count <= 0:
        return []
    
    if np is not None:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(items), size=keep_count, replace=False)
        indices = sorted(indices.tolist())
    else:
        import random
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(items)), k=keep_count))
    
    return [items[i] for i in indices]


def _filter_colmap_images_txt(
    images_txt_path: Path,
    keep_filenames: Set[str],
) -> None:
    """Rewrite COLMAP images.txt to only contain kept views."""
    lines = images_txt_path.read_text(encoding="utf-8").splitlines(keepends=False)
    
    out_lines: List[str] = []
    i = 0
    
    # Collect header/comment lines
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith("#")):
        out_lines.append(lines[i])
        i += 1
    
    # Parse image entries (2 lines per entry)
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("#"):
            out_lines.append(lines[i])
            i += 1
            continue
        
        image_line = lines[i]
        points2d_line = ""
        if i + 1 < len(lines):
            points2d_line = lines[i + 1]
        i += 2
        
        toks = image_line.strip().split()
        if len(toks) < 10:
            continue
        
        filename = os.path.basename(toks[-1])
        if filename in keep_filenames:
            out_lines.append(image_line)
            out_lines.append(points2d_line)
    
    images_txt_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _filter_dino_features(scene_sensor_root: Path, keep_filenames: Set[str]) -> None:
    """Filter DINO features to only contain kept views."""
    features_path = scene_sensor_root / "dino_features" / "features.pt"
    if not features_path.exists():
        return
    
    try:
        import torch
    except ImportError:
        print(f"[WARN] torch not available; skipping DINO features filter: {features_path}")
        return
    
    data = torch.load(str(features_path), map_location="cpu")
    if not isinstance(data, dict):
        print(f"[WARN] Unexpected DINO features format; skipping: {features_path}")
        return
    
    # Keep metadata keys and matching embeddings
    keep_stems = {Path(f).stem for f in keep_filenames}
    
    new_data = {}
    for key, value in data.items():
        if isinstance(key, str) and key.startswith("_"):
            # Metadata key
            new_data[key] = value
        elif key in keep_stems or key in keep_filenames:
            new_data[key] = value
    
    torch.save(new_data, str(features_path))
    print(f"Filtered DINO features: kept {len(new_data) - 2} embeddings -> {features_path}")


def _remove_unused_images(images_dir: Path, keep_filenames: Set[str]) -> int:
    """Remove image files not in keep set. Returns count of removed files."""
    removed = 0
    for img_path in images_dir.iterdir():
        if img_path.is_file() and img_path.name not in keep_filenames:
            img_path.unlink()
            removed += 1
    return removed


def _remove_unused_masks(masks_dir: Path, keep_filenames: Set[str]) -> int:
    """Remove mask files not corresponding to kept images. Returns count of removed files."""
    if not masks_dir.exists():
        return 0
    
    # Build set of expected mask names (same stem as images, but .png extension)
    keep_stems = {Path(f).stem for f in keep_filenames}
    
    removed = 0
    for mask_path in masks_dir.iterdir():
        if mask_path.is_file() and mask_path.stem not in keep_stems:
            mask_path.unlink()
            removed += 1
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy a ScanNet++ scene and sparsify it by subsampling training views."
    )
    
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/scenes/data",
        help="Root containing ScanNet++ scenes (default: data/scenes/data)",
    )
    parser.add_argument("--scene_id", type=str, required=True, help="Input scene id")
    parser.add_argument(
        "--out_scene_id",
        type=str,
        required=True,
        help="Output (copied) scene id to create under data_root",
    )
    parser.add_argument(
        "--sensor",
        type=str,
        default="dslr",
        help="Which sensor subfolder to sparsify (default: dslr)",
    )
    
    # Sparsification amount (mutually exclusive)
    amount_group = parser.add_mutually_exclusive_group(required=True)
    amount_group.add_argument(
        "--keep_fraction",
        type=float,
        help="Fraction of training views to keep (0.0 to 1.0)",
    )
    amount_group.add_argument(
        "--keep_count",
        type=int,
        help="Exact number of training views to keep",
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["every_n", "random"],
        default="every_n",
        help="Sampling strategy: 'every_n' (deterministic) or 'random' (default: every_n)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for 'random' strategy (default: 42)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If output scene already exists, delete it first",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would happen but don't write any files",
    )
    parser.add_argument(
        "--keep_all_images",
        action="store_true",
        help="Keep all image files (don't delete unused ones to save disk space)",
    )
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    in_scene_root = data_root / args.scene_id
    out_scene_root = data_root / args.out_scene_id
    
    in_sensor_root = in_scene_root / args.sensor
    out_sensor_root = out_scene_root / args.sensor
    
    if not in_sensor_root.exists():
        raise FileNotFoundError(f"Input sensor folder not found: {in_sensor_root}")
    
    transforms_path = in_sensor_root / "nerfstudio" / "transforms_undistorted.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"Missing transforms_undistorted.json: {transforms_path}")
    
    images_dir = in_sensor_root / "resized_undistorted_images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    
    masks_dir = in_sensor_root / "resized_undistorted_masks"
    has_masks = masks_dir.exists()
    
    colmap_images_txt = in_sensor_root / "colmap" / "images.txt"
    has_colmap = colmap_images_txt.exists()
    
    # Load transforms and build frame index
    transforms = _load_json(transforms_path)
    frame_index = _build_frame_index(transforms)
    
    # Determine train/test split
    split_lists = _split_from_train_test_lists(in_sensor_root)
    
    if split_lists is not None:
        train_files = [f for f in split_lists["train"] if f in frame_index]
        test_files = [f for f in split_lists["test"] if f in frame_index]
    else:
        # Use test_frames from transforms if available
        test_frames = transforms.get("test_frames", [])
        test_files_set = set()
        for fr in test_frames if isinstance(test_frames, list) else []:
            if isinstance(fr, dict) and fr.get("file_path"):
                test_files_set.add(os.path.basename(fr["file_path"]))
        
        train_files = [f for f in frame_index.keys() if f not in test_files_set]
        test_files = list(test_files_set)
    
    # Sort for deterministic ordering
    train_files = sorted(train_files)
    test_files = sorted(test_files)
    
    n_train_original = len(train_files)
    n_test = len(test_files)
    
    # Determine how many to keep
    if args.keep_fraction is not None:
        if not (0.0 < args.keep_fraction <= 1.0):
            raise ValueError("--keep_fraction must be between 0 and 1")
        keep_count = max(1, int(round(n_train_original * args.keep_fraction)))
    else:
        keep_count = args.keep_count
        if keep_count > n_train_original:
            print(f"[WARN] --keep_count ({keep_count}) > available train views ({n_train_original}), keeping all")
            keep_count = n_train_original
    
    # Select training views to keep
    if args.strategy == "every_n":
        # For every_n, compute the step to achieve approximately keep_count
        if keep_count >= n_train_original:
            kept_train = train_files
        else:
            step = max(1, n_train_original // keep_count)
            kept_train = train_files[::step]
            # Trim if we got more than requested
            if len(kept_train) > keep_count:
                kept_train = kept_train[:keep_count]
    else:  # random
        kept_train = _select_random(train_files, keep_count, args.seed)
    
    kept_train = sorted(kept_train)  # Keep sorted for consistency
    
    # All files to keep (train + test)
    all_keep = set(kept_train) | set(test_files)
    
    achieved_fraction = len(kept_train) / n_train_original if n_train_original > 0 else 0.0
    
    print(f"Input:  {in_sensor_root}")
    print(f"Output: {out_sensor_root}")
    print(f"Original train views: {n_train_original}")
    print(f"Test views (preserved): {n_test}")
    print(f"Strategy: {args.strategy}")
    print(f"Keeping {len(kept_train)} training views ({achieved_fraction:.1%})")
    print(f"Total views after sparsification: {len(all_keep)}")
    
    if args.dry_run:
        print("\nDry run: no files written.")
        print(f"\nKept training views would be:")
        for f in kept_train[:10]:
            print(f"  {f}")
        if len(kept_train) > 10:
            print(f"  ... and {len(kept_train) - 10} more")
        return 0
    
    # Check if output exists
    if out_scene_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output scene already exists: {out_scene_root} (use --overwrite)")
        shutil.rmtree(out_scene_root)
    
    # Copy the entire sensor folder
    print(f"\nCopying {in_sensor_root} -> {out_sensor_root}")
    _safe_mkdir(out_scene_root)
    shutil.copytree(in_sensor_root, out_sensor_root)
    
    # Update paths for output
    out_transforms_path = out_sensor_root / "nerfstudio" / "transforms_undistorted.json"
    out_images_dir = out_sensor_root / "resized_undistorted_images"
    out_masks_dir = out_sensor_root / "resized_undistorted_masks" if has_masks else None
    out_colmap_images_txt = out_sensor_root / "colmap" / "images.txt" if has_colmap else None
    
    # Update transforms JSON
    out_transforms = _load_json(out_transforms_path)
    
    # Filter frames to only kept ones
    frames = out_transforms.get("frames", [])
    out_transforms["frames"] = [
        fr for fr in frames
        if isinstance(fr, dict) and os.path.basename(fr.get("file_path", "")) in kept_train
    ]
    
    # test_frames should already be correct, but let's ensure it
    test_frames = out_transforms.get("test_frames", [])
    if isinstance(test_frames, list):
        out_transforms["test_frames"] = [
            fr for fr in test_frames
            if isinstance(fr, dict) and os.path.basename(fr.get("file_path", "")) in test_files
        ]
    
    _write_json(out_transforms_path, out_transforms)
    print(f"Updated: {out_transforms_path}")
    print(f"  frames: {len(out_transforms['frames'])}, test_frames: {len(out_transforms.get('test_frames', []))}")
    
    # Update train_test_lists.json if present
    train_test_path = _get_train_test_lists_path(out_sensor_root)
    if train_test_path:
        train_test_data = _load_json(train_test_path)
        train_test_data["train"] = kept_train
        # test stays the same
        _write_json(train_test_path, train_test_data)
        print(f"Updated: {train_test_path}")
    
    # Update COLMAP images.txt
    if out_colmap_images_txt and out_colmap_images_txt.exists():
        _filter_colmap_images_txt(out_colmap_images_txt, all_keep)
        print(f"Filtered: {out_colmap_images_txt}")
    
    # Filter DINO features
    _filter_dino_features(out_sensor_root, all_keep)
    
    # Remove unused image files (optional)
    if not args.keep_all_images:
        removed_images = _remove_unused_images(out_images_dir, all_keep)
        print(f"Removed {removed_images} unused images from {out_images_dir}")
        
        if out_masks_dir and out_masks_dir.exists():
            removed_masks = _remove_unused_masks(out_masks_dir, all_keep)
            print(f"Removed {removed_masks} unused masks from {out_masks_dir}")
    
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
