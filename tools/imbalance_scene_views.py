#!/usr/bin/env python3
"""Create an imbalanced copy of a scene by duplicating view(s).

Supports two dataset formats:
  - ScanNet++: <scene>/<sensor>/nerfstudio/transforms_undistorted.json
  - MipNeRF-360 / COLMAP: <scene>/sparse/0/ + <scene>/images/

Motivation
----------
For sanity-check ablations, it's useful to make the training set *artificially imbalanced*
without changing any poses/metadata. This script:

1) Copies a scene (so the original data is never modified)
2) Duplicates one or more existing views (image + mask)
3) Updates the nerfstudio transforms JSON and COLMAP `images.txt`/`images.bin` so the new views are
   treated as valid cameras (same extrinsics, new filenames)
4) (Optional) Updates DINO `features.pt` so selectors that rely on embeddings keep working

ScanNet++ layout:
  <data_root>/<scene_id>/dslr/
    nerfstudio/transforms_undistorted.json
    resized_undistorted_images/<filename>.JPG
    resized_undistorted_masks/<filename>.png
    colmap/images.txt

MipNeRF-360 / COLMAP layout:
  <data_root>/<scene_id>/
    images/<filename>.JPG
    sparse/0/images.bin, cameras.bin, points3D.bin

Examples
--------
ScanNet++ - Duplicate one training image until it accounts for ~30% of all train views:

  python tools/imbalance_scene_views.py \
    --data_root /path/to/scannetpp \
    --scene_id test_scene \
    --out_scene_id test_scene_imb_30 \
    --sensor dslr \
    --subset_fraction 0.30 \
    --focus DSC03721.JPG

MipNeRF-360 - Duplicate random training views until 80% imbalanced:

  python tools/imbalance_scene_views.py \
    --data_root /path/to/mipnerf360 \
    --scene_id bicycle \
    --out_scene_id bicycle_imb_80 \
    --format colmap \
    --subset_fraction 0.80 \
    --random_focus_count 1

Notes
-----
- The resulting achieved fraction is approximate because the number of duplicates must be integer.
- By default, if `<scene_root>/train_test_lists.json` exists, the script writes explicit
  `test_frames` into the transforms JSON to preserve a stable split.
- If you don't pass `--focus`, the script can randomly pick one (or N) focus view(s)
    from the *train* split (recommended) so the test set is unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
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


def _write_colmap_images_bin(path: Path, images: Dict[int, dict]) -> None:
    """Write COLMAP images.bin file."""
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(images)))
        for image_id, img in images.items():
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
#                          DATA CLASSES
# ============================================================================

@dataclass(frozen=True)
class DupSpec:
    src_file: str  # e.g. "DSC03721.JPG"
    split: str  # "train" or "test"


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _copytree(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dst} (pass --overwrite to replace)")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def _normalize_focus_names(focus: Iterable[str]) -> List[str]:
    # Keep original casing but strip any directory components.
    out: List[str] = []
    for x in focus:
        x = os.path.basename(x)
        if not x:
            continue
        out.append(x)
    return out


def _split_from_train_test_lists(scene_sensor_root: Path) -> Optional[Dict[str, List[str]]]:
    # Some datasets use train_test_lists.json; others use train_test_list.json.
    candidates = [
        scene_sensor_root / "train_test_lists.json",
        scene_sensor_root / "train_test_list.json",
    ]
    path = next((p for p in candidates if p.exists()), None)
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


def _get_train_test_lists_path(scene_sensor_root: Path) -> Optional[Path]:
    """Return the path to train_test_lists.json (or train_test_list.json) if it exists."""
    candidates = [
        scene_sensor_root / "train_test_lists.json",
        scene_sensor_root / "train_test_list.json",
    ]
    return next((p for p in candidates if p.exists()), None)


def _update_train_test_lists(
    scene_sensor_root: Path,
    dup_mappings_by_split: Dict[str, List[str]],
) -> None:
    """Update train_test_lists.json with duplicated image filenames.
    
    Args:
        scene_sensor_root: Path to the scene's sensor folder (e.g., <scene>/dslr)
        dup_mappings_by_split: Dict mapping split name ("train"/"test") to list of new filenames
    """
    path = _get_train_test_lists_path(scene_sensor_root)
    if path is None:
        return
    
    data = _load_json(path)
    if not isinstance(data, dict):
        return
    
    # Add duplicates to the appropriate split
    for split_name, new_filenames in dup_mappings_by_split.items():
        if split_name in data and isinstance(data[split_name], list):
            data[split_name].extend(new_filenames)
    
    _write_json(path, data)


def _build_frame_index(transforms: dict, include_test_frames: bool = True) -> Dict[str, dict]:
    """Build an index of all frames by filename.
    
    Args:
        transforms: The transforms dict containing frames and optionally test_frames
        include_test_frames: If True, also index frames from test_frames list
    """
    frames = transforms.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError("transforms['frames'] is not a list")

    index: Dict[str, dict] = {}
    for fr in frames:
        if not isinstance(fr, dict):
            continue
        fp = fr.get("file_path")
        if not fp:
            continue
        index[os.path.basename(fp)] = fr
    
    # Also index test_frames if present and requested
    if include_test_frames:
        test_frames = transforms.get("test_frames", [])
        if isinstance(test_frames, list):
            for fr in test_frames:
                if not isinstance(fr, dict):
                    continue
                fp = fr.get("file_path")
                if not fp:
                    continue
                # Don't overwrite if already in index (train takes precedence)
                basename = os.path.basename(fp)
                if basename not in index:
                    index[basename] = fr
    
    return index


def _ensure_transforms_have_test_frames(
    transforms: dict,
    split_lists: Dict[str, List[str]],
) -> dict:
    """Rewrite transforms to have explicit `frames` (train) and `test_frames` (test)."""

    frame_index = _build_frame_index(transforms)

    train_set = [f for f in split_lists["train"] if f in frame_index]
    test_set = [f for f in split_lists["test"] if f in frame_index]

    # Some scenes may have frames not mentioned by train_test_lists. Keep them in train by default.
    mentioned = set(train_set) | set(test_set)
    extras = [f for f in sorted(frame_index.keys()) if f not in mentioned]

    transforms = deepcopy(transforms)
    transforms["frames"] = [deepcopy(frame_index[f]) for f in (train_set + extras)]
    transforms["test_frames"] = [deepcopy(frame_index[f]) for f in test_set]
    return transforms


def _ensure_explicit_test_split(
    transforms: dict,
    test_count: int,
    seed: int,
) -> dict:
    """Ensure transforms contains a non-empty explicit `test_frames` split.

    If the file already has a non-empty `test_frames`, it is left unchanged.
    If `test_frames` is missing or empty, deterministically samples `test_count`
    frames from `frames` (pre-duplication), and moves them into `test_frames`.

    This keeps evaluation stable and prevents newly duplicated views from being
    accidentally selected into the test set.
    """

    transforms = deepcopy(transforms)
    frames = transforms.get("frames", [])
    test_frames = transforms.get("test_frames", None)

    if isinstance(test_frames, list) and len(test_frames) > 0:
        return transforms

    if not isinstance(frames, list) or len(frames) == 0:
        return transforms

    n_test = min(int(test_count), len(frames))
    if n_test <= 0:
        return transforms

    # Match dataset.py behavior (numpy default_rng) if numpy is available.
    if np is not None:
        rng = np.random.default_rng(int(seed))
        sample_indices = rng.choice(len(frames), size=n_test, replace=False)
        sample_indices = sorted(sample_indices.tolist())
    else:  # pragma: no cover
        import random

        rng = random.Random(int(seed))
        sample_indices = sorted(rng.sample(range(len(frames)), k=n_test))

    selected = set(sample_indices)
    transforms["test_frames"] = [deepcopy(frames[idx]) for idx in sample_indices]
    transforms["frames"] = [deepcopy(fr) for idx, fr in enumerate(frames) if idx not in selected]
    return transforms


def _compute_needed_duplicates(n_total: int, n_focus: int, target_fraction: float) -> int:
    """Compute number of *additional* copies needed so focus subset reaches target_fraction.

    Uses: (n_focus + m) / (n_total + m) ~= target_fraction
      => m = (target_fraction * n_total - n_focus) / (1 - target_fraction)

    Returns an integer m >= 0 (ceil to reach at least target_fraction).
    """

    if not (0.0 < target_fraction < 1.0):
        raise ValueError("--subset_fraction must be strictly between 0 and 1")
    if n_total <= 0:
        raise ValueError("n_total must be positive")
    if n_focus <= 0:
        raise ValueError("n_focus must be positive")
    if n_focus > n_total:
        raise ValueError("n_focus cannot exceed n_total")

    raw = (target_fraction * n_total - n_focus) / (1.0 - target_fraction)
    m = int(math.ceil(raw))
    return max(0, m)


def _make_dup_name(src_name: str, dup_idx: int) -> str:
    p = Path(src_name)
    return f"{p.stem}__dup{dup_idx:04d}{p.suffix}"


def _copy_view_files(
    images_dir: Path,
    masks_dir: Optional[Path],
    src_img_name: str,
    dst_img_name: str,
    src_mask_name: Optional[str],
    dst_mask_name: Optional[str],
) -> None:
    shutil.copy2(images_dir / src_img_name, images_dir / dst_img_name)

    if masks_dir is not None and src_mask_name and dst_mask_name:
        src_mask_path = masks_dir / src_mask_name
        if src_mask_path.exists():
            shutil.copy2(src_mask_path, masks_dir / dst_mask_name)


def _update_transforms_with_duplicates(
    transforms: dict,
    dup_frames_by_split: Dict[str, List[dict]],
) -> dict:
    transforms = deepcopy(transforms)

    # Default schema: everything is in frames.
    if "test_frames" not in transforms or not isinstance(transforms.get("test_frames"), list):
        transforms.setdefault("frames", [])
        transforms["frames"].extend(dup_frames_by_split.get("train", []))
        transforms["frames"].extend(dup_frames_by_split.get("test", []))
        return transforms

    transforms.setdefault("frames", [])
    transforms.setdefault("test_frames", [])
    transforms["frames"].extend(dup_frames_by_split.get("train", []))
    transforms["test_frames"].extend(dup_frames_by_split.get("test", []))
    return transforms


def _read_colmap_images_txt_pairs(path: Path) -> Tuple[List[str], Dict[str, Tuple[str, str]], int]:
    """Return (header_lines, name->(image_line, points2d_line), max_image_id)."""

    lines = path.read_text(encoding="utf-8").splitlines(keepends=False)

    header: List[str] = []
    pairs: Dict[str, Tuple[str, str]] = {}
    max_id = 0

    i = 0
    # Collect header/comment lines
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith("#")):
        header.append(lines[i])
        i += 1

    # Parse image entries (2 lines per entry)
    while i < len(lines):
        line = lines[i].strip("\n")
        if not line.strip():
            i += 1
            continue
        if line.lstrip().startswith("#"):
            header.append(lines[i])
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
        try:
            image_id = int(toks[0])
            max_id = max(max_id, image_id)
        except Exception:
            pass

        name = os.path.basename(toks[-1])
        pairs[name] = (image_line, points2d_line)

    return header, pairs, max_id


def _append_colmap_duplicates(
    images_txt_path: Path,
    duplicates: List[Tuple[str, str]],
) -> None:
    """Append duplicates to colmap/images.txt.

    duplicates: list of (src_name, dst_name)
    """

    header, pairs, max_id = _read_colmap_images_txt_pairs(images_txt_path)

    out_lines: List[str] = []
    # Reconstruct original file with identical content ordering
    # Keep original file content, then append new pairs.
    # (We avoid trying to preserve comments interspersed later; current files are header-only.)
    original_text = images_txt_path.read_text(encoding="utf-8").splitlines(keepends=False)
    out_lines.extend(original_text)

    next_id = max_id + 1

    for src_name, dst_name in duplicates:
        if src_name not in pairs:
            raise KeyError(
                f"Could not find COLMAP extrinsics for {src_name} in {images_txt_path}. "
                "(Needed to keep poses consistent.)"
            )
        image_line, points2d_line = pairs[src_name]
        toks = image_line.split()
        # Replace IMAGE_ID (first token) and NAME (last token)
        toks[0] = str(next_id)
        toks[-1] = dst_name
        new_image_line = " ".join(toks)
        out_lines.append(new_image_line)
        out_lines.append(points2d_line)
        next_id += 1

    images_txt_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _try_update_dino_features(scene_sensor_root: Path, mappings: List[Tuple[str, str]]) -> None:
    """Best-effort update of <scene_root>/dino_features/features.pt.

    Adds embeddings for duplicated image stems to match their source stems.
    """

    features_path = scene_sensor_root / "dino_features" / "features.pt"
    if not features_path.exists():
        return

    try:
        import torch  # type: ignore
    except Exception:
        print(f"[WARN] torch not available; skipping DINO features update: {features_path}")
        return

    data = torch.load(str(features_path), map_location="cpu")
    if not isinstance(data, dict):
        print(f"[WARN] Unexpected DINO features format (not a dict); skipping: {features_path}")
        return

    # Preserve metadata keys
    meta_keys = {k for k in data.keys() if isinstance(k, str) and k.startswith("_")}

    added = 0
    for src_name, dst_name in mappings:
        src_stem = Path(src_name).stem
        dst_stem = Path(dst_name).stem

        if dst_stem in data:
            continue

        if src_stem in data:
            data[dst_stem] = data[src_stem]
            added += 1
            continue

        # Some extraction pipelines might key by full filename; try that too.
        if src_name in data:
            data[dst_stem] = data[src_name]
            added += 1
            continue

        # Otherwise, skip (selector has fuzzy matching fallback).

    if added > 0:
        torch.save(data, str(features_path))
        print(f"Updated DINO features: added {added} embeddings -> {features_path}")


# ============================================================================
#                          COLMAP / MIPNERF-360 IMBALANCING
# ============================================================================

def _imbalance_colmap_scene(args, in_scene_root: Path, out_scene_root: Path) -> int:
    """Imbalance a COLMAP/MipNeRF-360 format scene by duplicating views."""
    import random
    
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
    
    # Determine focus views
    focus = _normalize_focus_names(args.focus) if args.focus else []
    
    if args.random_focus_count is not None:
        random.seed(args.random_seed)
        n_pick = min(args.random_focus_count, len(train_files))
        focus = random.sample(train_files, n_pick)
        print(f"Randomly selected {len(focus)} focus view(s) from train split: {focus[:5]}{'...' if len(focus) > 5 else ''}")
    
    if not focus:
        # Default: pick one random training view
        random.seed(args.random_seed)
        focus = random.sample(train_files, 1)
        print(f"No focus specified; randomly selected: {focus}")
    
    # Validate focus views are in train set
    focus_set = set(focus)
    invalid = focus_set - set(train_files)
    if invalid:
        raise ValueError(f"Focus view(s) not in training set: {invalid}")
    
    # Calculate duplicates needed
    subset_fraction = args.subset_fraction
    if not (0.0 < subset_fraction < 1.0):
        raise ValueError("--subset_fraction must be between 0 and 1 (exclusive)")
    
    # After imbalancing: focus_count + n_dups = subset_fraction * (n_train + n_dups)
    # => n_dups = (subset_fraction * n_train - focus_count) / (1 - subset_fraction)
    focus_count = len(focus)
    n_dups_needed = math.ceil((subset_fraction * n_train_original - focus_count) / (1 - subset_fraction))
    n_dups_needed = max(0, n_dups_needed)
    
    if n_dups_needed == 0:
        print(f"[WARN] Focus views already exceed target fraction; no duplicates needed.")
        return 0
    
    # Distribute duplicates round-robin across focus views
    dup_list: List[Tuple[str, str]] = []
    for i in range(n_dups_needed):
        src = focus[i % len(focus)]
        src_stem = Path(src).stem
        src_ext = Path(src).suffix
        dst_name = f"{src_stem}_dup{i+1}{src_ext}"
        dup_list.append((src, dst_name))
    
    n_final_train = n_train_original + n_dups_needed
    achieved_fraction = (focus_count + n_dups_needed) / n_final_train
    
    print(f"Input:  {in_scene_root}")
    print(f"Output: {out_scene_root}")
    print(f"Original train views: {n_train_original}")
    print(f"Test views (preserved): {n_test}")
    print(f"LLFF-hold: {args.llffhold}")
    print(f"Focus views: {focus_count} ({focus[:3]}{'...' if len(focus) > 3 else ''})")
    print(f"Duplicates to add: {n_dups_needed}")
    print(f"Final train views: {n_final_train}")
    print(f"Achieved imbalance: {achieved_fraction:.1%}")
    
    if args.dry_run:
        print("\nDry run: no files written.")
        print(f"\nFirst 10 duplicates would be:")
        for src, dst in dup_list[:10]:
            print(f"  {src} -> {dst}")
        if len(dup_list) > 10:
            print(f"  ... and {len(dup_list) - 10} more")
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
    
    # Copy duplicate images to the selected image folders only
    for folder_name in image_folders_to_copy:
        out_images_dir = out_scene_root / folder_name
        if out_images_dir.exists():
            dups_created = 0
            for src_name, dst_name in dup_list:
                src_path = out_images_dir / src_name
                dst_path = out_images_dir / dst_name
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                    dups_created += 1
            if dups_created > 0:
                print(f"Created {dups_created} duplicate images in {out_images_dir}")
    
    # Update COLMAP images.bin
    out_images_bin = out_scene_root / "sparse" / "0" / "images.bin"
    max_id = max(images_dict.keys())
    
    # Add duplicate entries
    new_images_dict = dict(images_dict)
    for i, (src_name, dst_name) in enumerate(dup_list):
        # Find source image entry
        src_entry = next((img for img in images_dict.values() if img["name"] == src_name), None)
        if src_entry is None:
            print(f"[WARN] Could not find COLMAP entry for {src_name}, skipping")
            continue
        
        new_id = max_id + i + 1
        new_images_dict[new_id] = {
            "id": new_id,
            "name": dst_name,
            "qvec": src_entry["qvec"],
            "tvec": src_entry["tvec"],
            "camera_id": src_entry["camera_id"],
        }
    
    _write_colmap_images_bin(out_images_bin, new_images_dict)
    print(f"Updated: {out_images_bin} ({len(new_images_dict)} images)")
    
    # Update DINO features if present (always, to keep consistency)
    dino_dir = out_scene_root / "dino_features"
    if dino_dir.exists():
        _try_update_dino_features(out_scene_root, dup_list)
    
    print("\nDone (COLMAP format).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy a scene and imbalance it by duplicating selected view(s) while keeping poses. "
                    "Supports ScanNet++ and MipNeRF-360/COLMAP formats."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=str(Path("data/scenes/data")),
        help="Root containing scenes",
    )
    parser.add_argument("--scene_id", type=str, required=True, help="Input scene id (folder name under data_root)")
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
    parser.add_argument(
        "--subset_fraction",
        type=float,
        required=True,
        help="Target fraction of final dataset to be the focus subset (0..1, exclusive)",
    )

    focus_group = parser.add_mutually_exclusive_group(required=False)
    focus_group.add_argument(
        "--focus",
        nargs="+",
        default=None,
        help="One or more existing view filenames to duplicate (e.g. DSC03721.JPG)",
    )
    focus_group.add_argument(
        "--random_focus_count",
        type=int,
        default=None,
        help="If set, randomly pick this many focus views from the TRAIN split (recommended)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Seed for random focus selection (default: 0)",
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
        "--write_test_frames",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If train_test_lists.json exists, write explicit test_frames into transforms JSON (default: true)",
    )
    parser.add_argument(
        "--ensure_test_frames",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If transforms has no non-empty test_frames, create a deterministic test split before duplicating (default: true)",
    )
    parser.add_argument(
        "--test_count",
        type=int,
        default=10,
        help="If --ensure_test_frames is enabled, number of test frames to create (default: 10)",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help="Seed for deterministic test split creation (default: 0, matches dataset.py)",
    )
    parser.add_argument(
        "--update_dino_features",
        action="store_true",
        help="Best-effort update dino_features/features.pt by copying embeddings for duplicates",
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
        return _imbalance_colmap_scene(args, in_scene_root, out_scene_root)
    
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
    if not colmap_images_txt.exists():
        raise FileNotFoundError(f"Missing COLMAP extrinsics file: {colmap_images_txt}")

    focus = _normalize_focus_names(args.focus) if args.focus else []

    transforms = _load_json(transforms_path)

    split_lists = _split_from_train_test_lists(in_sensor_root)

    # Optionally rewrite transforms to have explicit train/test frames based on train_test_list(s).json
    if args.write_test_frames and split_lists is not None:
        transforms = _ensure_transforms_have_test_frames(transforms, split_lists)

    # If we still don't have a non-empty explicit test split, create one *before* duplicating.
    if args.ensure_test_frames:
        transforms = _ensure_explicit_test_split(transforms, test_count=args.test_count, seed=args.split_seed)

    frames = transforms.get("frames", [])
    test_frames = transforms.get("test_frames", []) if isinstance(transforms.get("test_frames"), list) else []

    def _frame_list_to_map(lst: list) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for fr in lst:
            if not isinstance(fr, dict):
                continue
            fp = fr.get("file_path")
            if fp:
                out[os.path.basename(fp)] = fr
        return out

    train_map = _frame_list_to_map(frames)
    test_map = _frame_list_to_map(test_frames)

    # If no explicit focus was given, optionally choose focus views randomly from TRAIN.
    if not focus:
        k = int(args.random_focus_count) if args.random_focus_count is not None else 1
        if k <= 0:
            raise ValueError("--random_focus_count must be >= 1")

        # Prefer the explicit split file, since that's what you usually want to preserve.
        if split_lists is not None and isinstance(split_lists.get("train"), list) and len(split_lists["train"]) > 0:
            train_candidates = [f for f in split_lists["train"] if f in train_map]
        else:
            train_candidates = sorted(train_map.keys())

        if len(train_candidates) == 0:
            raise FileNotFoundError(
                "No train candidates available to sample focus views from. "
                "(Check transforms_undistorted.json and resized_undistorted_images.)"
            )

        if k > len(train_candidates):
            raise ValueError(f"Requested {k} random focus views but only {len(train_candidates)} train candidates exist")

        # Deterministic RNG for reproducibility.
        if np is not None:
            rng = np.random.default_rng(int(args.random_seed))
            idxs = rng.choice(len(train_candidates), size=k, replace=False)
            focus = [train_candidates[i] for i in sorted(idxs.tolist())]
        else:  # pragma: no cover
            import random

            rng = random.Random(int(args.random_seed))
            focus = rng.sample(train_candidates, k=k)
            focus = sorted(focus)

    # Determine which split each focus frame belongs to.
    dup_specs: List[DupSpec] = []
    for name in focus:
        if name in train_map:
            dup_specs.append(DupSpec(src_file=name, split="train"))
        elif name in test_map:
            # We allow explicitly targeting test frames, but discourage it.
            dup_specs.append(DupSpec(src_file=name, split="test"))
        else:
            raise KeyError(
                f"Focus view not found in transforms: {name}. "
                "Check it exists in transforms_undistorted.json frames/test_frames."
            )

    # Total dataset size is train+test as represented in transforms.
    n_total = len(train_map) + len(test_map)
    n_focus = len(dup_specs)

    m_total = _compute_needed_duplicates(n_total=n_total, n_focus=n_focus, target_fraction=float(args.subset_fraction))

    achieved = (n_focus + m_total) / (n_total + m_total) if (n_total + m_total) > 0 else 0.0

    print(f"Input:  {in_sensor_root}")
    print(f"Output: {out_sensor_root}")
    print(f"Total views (before): {n_total} (train={len(train_map)}, test={len(test_map)})")
    print(f"Focus views: {n_focus}: {[d.src_file for d in dup_specs]}")
    print(f"Target subset_fraction: {args.subset_fraction:.4f}")
    print(f"Planned duplicates to add: {m_total}")
    print(f"Achieved subset_fraction: {achieved:.4f}")

    if args.dry_run:
        print("Dry run: no files written.")
        return 0

    # Copy scene (sensor-only) to output
    _safe_mkdir(out_scene_root)
    _copytree(in_sensor_root, out_sensor_root, overwrite=args.overwrite)

    # Recompute paths in output
    out_transforms_path = out_sensor_root / "nerfstudio" / "transforms_undistorted.json"
    out_images_dir = out_sensor_root / "resized_undistorted_images"
    out_masks_dir = out_sensor_root / "resized_undistorted_masks" if has_masks else None
    out_colmap_images_txt = out_sensor_root / "colmap" / "images.txt"

    out_transforms = _load_json(out_transforms_path)

    # If we rewrote in-memory transforms for splitting, apply that to output too.
    if args.write_test_frames and split_lists is not None:
        out_transforms = _ensure_transforms_have_test_frames(out_transforms, split_lists)

    if args.ensure_test_frames:
        out_transforms = _ensure_explicit_test_split(out_transforms, test_count=args.test_count, seed=args.split_seed)

    out_frames = out_transforms.get("frames", [])
    out_test_frames = out_transforms.get("test_frames", []) if isinstance(out_transforms.get("test_frames"), list) else []

    out_train_map = _frame_list_to_map(out_frames)
    out_test_map = _frame_list_to_map(out_test_frames)

    # Build a round-robin schedule of which focus view to duplicate
    schedule: List[DupSpec] = []
    for i in range(m_total):
        schedule.append(dup_specs[i % len(dup_specs)])

    # Prepare duplicates and apply
    dup_mappings: List[Tuple[str, str]] = []  # (src_name, dst_name)
    dup_frames_by_split: Dict[str, List[dict]] = {"train": [], "test": []}
    dup_filenames_by_split: Dict[str, List[str]] = {"train": [], "test": []}  # For train_test_lists.json

    dup_counter_by_src: Dict[str, int] = {d.src_file: 0 for d in dup_specs}

    for spec in schedule:
        dup_counter_by_src[spec.src_file] += 1
        dst_name = _make_dup_name(spec.src_file, dup_counter_by_src[spec.src_file])

        # Ensure no collision (rare unless rerun without overwrite)
        while (out_images_dir / dst_name).exists():
            dup_counter_by_src[spec.src_file] += 1
            dst_name = _make_dup_name(spec.src_file, dup_counter_by_src[spec.src_file])

        if spec.split == "train":
            src_frame = out_train_map[spec.src_file]
        else:
            src_frame = out_test_map[spec.src_file]

        new_frame = deepcopy(src_frame)
        new_frame["file_path"] = dst_name

        src_mask = new_frame.get("mask_path", None)
        dst_mask = None
        if isinstance(src_mask, str) and src_mask:
            src_mask_base = os.path.basename(src_mask)
            # Keep mask extension (.png) but align stem with duplicated image
            dst_mask = f"{Path(dst_name).stem}{Path(src_mask_base).suffix}"
            new_frame["mask_path"] = dst_mask

        _copy_view_files(
            images_dir=out_images_dir,
            masks_dir=out_masks_dir,
            src_img_name=spec.src_file,
            dst_img_name=dst_name,
            src_mask_name=os.path.basename(src_mask) if isinstance(src_mask, str) else None,
            dst_mask_name=dst_mask,
        )

        dup_mappings.append((spec.src_file, dst_name))
        dup_frames_by_split[spec.split].append(new_frame)
        dup_filenames_by_split[spec.split].append(dst_name)

    # Update transforms JSON with new frames
    out_transforms = _update_transforms_with_duplicates(out_transforms, dup_frames_by_split)
    _write_json(out_transforms_path, out_transforms)

    # Update COLMAP extrinsics to include duplicates
    _append_colmap_duplicates(out_colmap_images_txt, dup_mappings)

    # Update train_test_lists.json with duplicated filenames
    _update_train_test_lists(out_sensor_root, dup_filenames_by_split)
    train_test_lists_path = _get_train_test_lists_path(out_sensor_root)

    # Optionally update DINO features
    if args.update_dino_features:
        _try_update_dino_features(out_sensor_root, dup_mappings)

    print("Done.")
    print(f"Wrote: {out_transforms_path}")
    print(f"Updated: {out_colmap_images_txt}")
    if train_test_lists_path:
        print(f"Updated: {train_test_lists_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
