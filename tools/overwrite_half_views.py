#!/usr/bin/env python3
"""Overwrite a fraction of a scene's images with a single image (keeping filenames).

This is useful for controlled ablations where you want poses/metadata to remain
valid (same filenames), but pixel content changes.

Example:
  python tools/overwrite_half_views.py \
    --source_image /home/fgrob/garching_wednesday.jpeg \
    --transforms_json /home/fgrob/data/scenes/data/test_scene/dslr/nerfstudio/transforms_undistorted.json \
    --target_dir /home/fgrob/data/scenes/data/test_scene/dslr/resized_undistorted_images \
    --fraction 0.5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageOps


def _load_target_size(transforms_json: Path | None, target_dir: Path) -> Tuple[int, int]:
    if transforms_json is not None:
        with transforms_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
        w = int(data["w"])
        h = int(data["h"])
        return w, h

    # Fallback: read size from first image in directory
    candidates = sorted([p for p in target_dir.iterdir() if p.is_file()])
    if not candidates:
        raise FileNotFoundError(f"No files found in target_dir: {target_dir}")
    with Image.open(candidates[0]) as im:
        return im.size  # (w, h)


def _iter_target_images(target_dir: Path, exts: Iterable[str]) -> list[Path]:
    exts_norm = {e.lower().lstrip(".") for e in exts}
    out: list[Path] = []
    for p in target_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower().lstrip(".") in exts_norm:
            out.append(p)
    return sorted(out, key=lambda x: x.name)


def _overwrite_one(src: Image.Image, target_path: Path, size: Tuple[int, int], quality: int) -> None:
    # Make orientation consistent, then resize exactly to required size.
    im = ImageOps.exif_transpose(src)
    if im.mode != "RGB":
        im = im.convert("RGB")
    if im.size != size:
        im = im.resize(size, resample=Image.BICUBIC)

    # Save as JPEG regardless of original extension, but preserve filename.
    im.save(target_path, format="JPEG", quality=quality, optimize=True, subsampling=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Overwrite a fraction of views with a single image, keeping filenames")
    parser.add_argument("--source_image", type=str, required=True, help="Path to the source image to paste everywhere")
    parser.add_argument("--target_dir", type=str, required=True, help="Directory containing the existing views to overwrite")
    parser.add_argument(
        "--transforms_json",
        type=str,
        default=None,
        help="Optional nerfstudio transforms JSON to get required w/h (recommended)",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.5,
        help="Fraction of images to overwrite (0..1). Default: 0.5",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=["JPG", "JPEG", "PNG"],
        help="Which file extensions inside target_dir are eligible (case-insensitive)",
    )
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality for overwritten images")
    args = parser.parse_args()

    src_path = Path(args.source_image)
    target_dir = Path(args.target_dir)
    transforms_json = Path(args.transforms_json) if args.transforms_json else None

    if not src_path.exists():
        raise FileNotFoundError(f"source_image not found: {src_path}")
    if not target_dir.exists():
        raise FileNotFoundError(f"target_dir not found: {target_dir}")
    if transforms_json is not None and not transforms_json.exists():
        raise FileNotFoundError(f"transforms_json not found: {transforms_json}")
    if not (0.0 <= args.fraction <= 1.0):
        raise ValueError("--fraction must be between 0 and 1")

    target_images = _iter_target_images(target_dir, args.extensions)
    if not target_images:
        raise FileNotFoundError(
            f"No target images found in {target_dir} with extensions {args.extensions}"
        )

    size = _load_target_size(transforms_json, target_dir)
    n_total = len(target_images)
    n_overwrite = int(round(n_total * args.fraction))

    # Deterministic choice: first N files in sorted order.
    to_overwrite = target_images[:n_overwrite]

    print(f"Target dir: {target_dir}")
    print(f"Target size: {size[0]}x{size[1]}")
    print(f"Total images: {n_total}")
    print(f"Overwriting: {n_overwrite} ({args.fraction:.3f})")

    with Image.open(src_path) as src_im:
        for i, target_path in enumerate(to_overwrite, start=1):
            _overwrite_one(src_im, target_path, size=size, quality=args.quality)
            if i % 50 == 0 or i == n_overwrite:
                print(f"  wrote {i}/{n_overwrite}: {os.path.basename(target_path)}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
