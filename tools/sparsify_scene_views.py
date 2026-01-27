#!/usr/bin/env python3
"""Create a sparse copy of a scene by subsampling training views.

Supports two dataset formats:
  - ScanNet++: <scene>/<sensor>/nerfstudio/transforms_undistorted.json
  - MipNeRF-360 / COLMAP: <scene>/sparse/0/ + <scene>/images/

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
ScanNet++ - Keep 50 training images:

  python tools/sparsify_scene_views.py \
    --data_root /path/to/scannetpp \
    --scene_id test_scene \
    --out_scene_id test_scene_sparse_50 \
    --sensor dslr \
    --keep_count 50 \
    --strategy every_n

MipNeRF-360 - Keep 50 training images:

  python tools/sparsify_scene_views.py \
    --data_root /path/to/mipnerf360 \
    --scene_id bicycle \
    --out_scene_id bicycle_sparse_50 \
    --format colmap \
    --keep_count 50 \
    --strategy every_n

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


# ============================================================================
#                          FORMAT DETECTION
# ============================================================================

def detect_format(scene_root: Path, sensor: str = "dslr") -> str:
    """Detect the dataset format based on directory structure.
    
    Returns:
        'scannetpp' if ScanNet++ format is detected
        'colmap' if MipNeRF-360/COLMAP format is detected
    """
    # Check for ScanNet++ structure: <scene>/<sensor>/nerfstudio/
    scannetpp_path = scene_root / sensor / "nerfstudio"
    if scannetpp_path.exists():
        return "scannetpp"
    
    # Check for COLMAP structure: <scene>/sparse/0/
    colmap_path = scene_root / "sparse" / "0"
    if colmap_path.exists():
        return "colmap"
    
    raise ValueError(
        f"Could not detect dataset format for {scene_root}. "
        f"Expected either {scannetpp_path} or {colmap_path} to exist."
    )


def _read_colmap_images_bin(path: Path) -> Dict[int, dict]:
    """Read COLMAP images.bin file."""
    import struct
    
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("dddd", f.read(32))
            tx, ty, tz = struct.unpack("ddd", f.read(24))
            camera_id = struct.unpack("I", f.read(4))[0]
            
            # Read image name (null-terminated string)
            name_chars = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_chars.append(c.decode("utf-8"))
            name = "".join(name_chars)
            
            # Read 2D points (skip them)
            num_points2d = struct.unpack("Q", f.read(8))[0]
            f.read(24 * num_points2d)  # x, y, point3d_id per point
            
            images[image_id] = {
                "id": image_id,
                "name": name,
                "qvec": (qw, qx, qy, qz),
                "tvec": (tx, ty, tz),
                "camera_id": camera_id,
            }
    
    return images


def _write_colmap_images_bin(path: Path, images: Dict[int, dict], keep_names: Set[str]) -> None:
    """Write filtered COLMAP images.bin file."""
    import struct
    
    # Filter to only keep specified images
    kept_images = {k: v for k, v in images.items() if v["name"] in keep_names}
    
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(kept_images)))
        for image_id, img in kept_images.items():
            f.write(struct.pack("I", img["id"]))
            f.write(struct.pack("dddd", *img["qvec"]))
            f.write(struct.pack("ddd", *img["tvec"]))
            f.write(struct.pack("I", img["camera_id"]))
            f.write(img["name"].encode("utf-8") + b"\x00")
            f.write(struct.pack("Q", 0))  # No 2D points


def _get_colmap_train_test_split(all_names: List[str], llffhold: int = 8) -> Dict[str, List[str]]:
    """Split images using LLFF-hold convention (every Nth image for test)."""
    sorted_names = sorted(all_names)
    test_names = [name for i, name in enumerate(sorted_names) if i % llffhold == 0]
    train_names = [name for i, name in enumerate(sorted_names) if i % llffhold != 0]
    return {"train": train_names, "test": test_names}


# ============================================================================
#                          UTILITY FUNCTIONS  
# ============================================================================

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


# ============================================================================
#                          COLMAP / MIPNERF-360 SPARSIFICATION
# ============================================================================

def _sparsify_colmap_scene(args, in_scene_root: Path, out_scene_root: Path) -> int:
    """Sparsify a COLMAP/MipNeRF-360 format scene."""
    
    # Validate input paths
    colmap_dir = in_scene_root / "sparse" / "0"
    images_bin = colmap_dir / "images.bin"
    images_dir = in_scene_root / "images"
    
    if not colmap_dir.exists():
        raise FileNotFoundError(f"COLMAP sparse folder not found: {colmap_dir}")
    if not images_bin.exists():
        raise FileNotFoundError(f"COLMAP images.bin not found: {images_bin}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    
    # Read COLMAP images.bin
    images_dict = _read_colmap_images_bin(images_bin)
    all_names = [img["name"] for img in images_dict.values()]
    
    # Get train/test split using LLFF-hold convention
    split = _get_colmap_train_test_split(all_names, llffhold=args.llffhold)
    train_files = sorted(split["train"])
    test_files = sorted(split["test"])
    
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
        if keep_count >= n_train_original:
            kept_train = train_files
        else:
            step = max(1, n_train_original // keep_count)
            kept_train = train_files[::step]
            if len(kept_train) > keep_count:
                kept_train = kept_train[:keep_count]
    else:  # random
        kept_train = _select_random(train_files, keep_count, args.seed)
    
    kept_train = sorted(kept_train)
    
    # All files to keep (train + test)
    all_keep = set(kept_train) | set(test_files)
    
    achieved_fraction = len(kept_train) / n_train_original if n_train_original > 0 else 0.0
    
    print(f"Input:  {in_scene_root}")
    print(f"Output: {out_scene_root}")
    print(f"Original train views: {n_train_original}")
    print(f"Test views (preserved): {n_test}")
    print(f"Strategy: {args.strategy}")
    print(f"LLFF-hold: {args.llffhold}")
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
    
    # Determine which image folders to copy
    all_image_folders = ["images", "images_2", "images_4", "images_8"]
    images_arg = getattr(args, 'images', None)
    if images_arg:
        # Only copy the specified folder
        if not (in_scene_root / images_arg).exists():
            raise FileNotFoundError(f"Specified images folder not found: {in_scene_root / images_arg}")
        image_folders_to_copy = [images_arg]
        print(f"Images folder: {images_arg} (only)")
    else:
        # Copy all existing image folders
        image_folders_to_copy = [f for f in all_image_folders if (in_scene_root / f).exists()]
        print(f"Images folders: {image_folders_to_copy}")
    
    # Selective copy: copy structure but skip unwanted image folders
    print(f"\nCopying {in_scene_root} -> {out_scene_root}")
    os.makedirs(out_scene_root, exist_ok=True)
    
    for item in in_scene_root.iterdir():
        src = in_scene_root / item.name
        dst = out_scene_root / item.name
        
        # Skip image folders we don't want
        if item.name in all_image_folders and item.name not in image_folders_to_copy:
            continue
        
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    
    # Update COLMAP images.bin
    out_images_bin = out_scene_root / "sparse" / "0" / "images.bin"
    _write_colmap_images_bin(out_images_bin, images_dict, all_keep)
    print(f"Filtered: {out_images_bin} ({len(all_keep)} images)")
    
    # Filter DINO features if present
    dino_dir = out_scene_root / "dino_features"
    if dino_dir.exists():
        _filter_dino_features(out_scene_root, all_keep)
    
    # Remove unused images from copied image folders
    if not args.keep_all_images:
        for folder_name in image_folders_to_copy:
            out_images_dir = out_scene_root / folder_name
            if out_images_dir.exists():
                removed_images = _remove_unused_images(out_images_dir, all_keep)
                print(f"Removed {removed_images} unused images from {out_images_dir}")
    
    print("\nDone (COLMAP format).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy a scene and sparsify it by subsampling training views. "
                    "Supports ScanNet++ and MipNeRF-360/COLMAP formats."
    )
    
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/scenes/data",
        help="Root containing scenes",
    )
    parser.add_argument("--scene_id", type=str, required=True, help="Input scene id")
    parser.add_argument(
        "--out_scene_id",
        type=str,
        required=True,
        help="Output (copied) scene id to create under data_root",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["auto", "scannetpp", "colmap"],
        default="auto",
        help="Dataset format: 'auto' (detect), 'scannetpp', or 'colmap' (MipNeRF-360) (default: auto)",
    )
    parser.add_argument(
        "--sensor",
        type=str,
        default="dslr",
        help="Sensor subfolder for ScanNet++ format (default: dslr)",
    )
    parser.add_argument(
        "--llffhold",
        type=int,
        default=8,
        help="LLFF-hold value for COLMAP format (every Nth for test, default: 8)",
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
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="For COLMAP datasets: which images folder to keep (e.g., images_4). "
             "If not specified, keeps all image folders. When specified, only that folder is copied.",
    )
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    in_scene_root = data_root / args.scene_id
    out_scene_root = data_root / args.out_scene_id
    
    # Detect or use specified format
    if args.format == "auto":
        dataset_format = detect_format(in_scene_root, args.sensor)
        print(f"Auto-detected format: {dataset_format}")
    else:
        dataset_format = args.format
    
    # =========================================================================
    # COLMAP / MipNeRF-360 FORMAT
    # =========================================================================
    if dataset_format == "colmap":
        return _sparsify_colmap_scene(args, in_scene_root, out_scene_root)
    
    # =========================================================================
    # SCANNETPP FORMAT (original logic)
    # =========================================================================
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
