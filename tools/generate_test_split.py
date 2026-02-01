#!/usr/bin/env python3
"""Generate test.txt for COLMAP/MipNeRF-360 scenes.

This tool creates a stable test split file (sparse/0/test.txt) for COLMAP scenes.
The test.txt file ensures that all manipulated versions of a scene (sparse, imbalanced)
use the same test images as the baseline, enabling fair comparisons.

When test.txt exists:
- The 3DGS training code (scene/dataset_readers.py) will use it instead of LLFF-hold
- The manipulation scripts (sparsify, imbalance) will preserve the test images

Usage:
    # Generate test.txt for a single scene
    python tools/generate_test_split.py --scene_path /path/to/bicycle
    
    # Generate test.txt for multiple scenes
    python tools/generate_test_split.py --data_root /path/to/mipnerf360 --scenes bicycle garden stump
    
    # Custom LLFF-hold value
    python tools/generate_test_split.py --scene_path /path/to/bicycle --llffhold 4
    
    # Dry run (show what would be created)
    python tools/generate_test_split.py --scene_path /path/to/bicycle --dry_run
"""

import argparse
import os
import struct
from pathlib import Path
from typing import Dict, List


def read_colmap_images_bin(path: Path) -> Dict[int, dict]:
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


def get_llff_hold_split(all_names: List[str], llffhold: int = 8) -> Dict[str, List[str]]:
    """Split images using LLFF-hold convention (every Nth image for test)."""
    sorted_names = sorted(all_names)
    test_names = [name for i, name in enumerate(sorted_names) if i % llffhold == 0]
    train_names = [name for i, name in enumerate(sorted_names) if i % llffhold != 0]
    return {"train": train_names, "test": test_names}


def generate_test_txt(scene_path: Path, llffhold: int = 8, dry_run: bool = False, force: bool = False) -> bool:
    """Generate test.txt for a COLMAP scene.
    
    Args:
        scene_path: Path to the scene root directory
        llffhold: LLFF-hold value (every Nth image for test)
        dry_run: If True, just print what would be done
        force: If True, overwrite existing test.txt
        
    Returns:
        True if test.txt was created/already exists, False on error
    """
    colmap_dir = scene_path / "sparse" / "0"
    images_bin = colmap_dir / "images.bin"
    test_txt_path = colmap_dir / "test.txt"
    
    # Check if scene is COLMAP format
    if not colmap_dir.exists():
        print(f"[SKIP] Not a COLMAP scene (no sparse/0/): {scene_path}")
        return False
    
    if not images_bin.exists():
        print(f"[ERROR] images.bin not found: {images_bin}")
        return False
    
    # Check if test.txt already exists
    if test_txt_path.exists() and not force:
        with open(test_txt_path, 'r') as f:
            existing_count = len([line for line in f if line.strip()])
        print(f"[SKIP] test.txt already exists ({existing_count} test images): {test_txt_path}")
        return True
    
    # Read images
    images_dict = read_colmap_images_bin(images_bin)
    all_names = [img["name"] for img in images_dict.values()]
    
    # Compute split
    split = get_llff_hold_split(all_names, llffhold=llffhold)
    test_names = sorted(split["test"])
    train_names = sorted(split["train"])
    
    print(f"Scene: {scene_path}")
    print(f"  Total images: {len(all_names)}")
    print(f"  LLFF-hold: {llffhold}")
    print(f"  Test images: {len(test_names)}")
    print(f"  Train images: {len(train_names)}")
    
    if dry_run:
        print(f"  [DRY RUN] Would write test.txt with {len(test_names)} entries")
        print(f"  Test images preview: {test_names[:5]}...")
        return True
    
    # Write test.txt
    with open(test_txt_path, 'w') as f:
        for name in test_names:
            f.write(name + '\n')
    
    print(f"  [CREATED] {test_txt_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate test.txt for COLMAP/MipNeRF-360 scenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Scene specification (either single scene or batch)
    parser.add_argument("--scene_path", type=str, default=None,
                        help="Path to a single scene to process")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root directory containing multiple scenes")
    parser.add_argument("--scenes", type=str, nargs="+", default=None,
                        help="Scene names to process (used with --data_root)")
    
    # Options
    parser.add_argument("--llffhold", type=int, default=8,
                        help="LLFF-hold value (every Nth image for test). Default: 8")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be done without writing files")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing test.txt files")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.scene_path is None and args.data_root is None:
        parser.error("Either --scene_path or --data_root must be specified")
    
    if args.scene_path is not None and args.data_root is not None:
        parser.error("Cannot specify both --scene_path and --data_root")
    
    if args.data_root is not None and args.scenes is None:
        parser.error("--scenes must be specified when using --data_root")
    
    # Process scenes
    success_count = 0
    fail_count = 0
    
    if args.scene_path:
        # Single scene
        scene_path = Path(args.scene_path)
        if generate_test_txt(scene_path, llffhold=args.llffhold, dry_run=args.dry_run, force=args.force):
            success_count += 1
        else:
            fail_count += 1
    else:
        # Multiple scenes
        data_root = Path(args.data_root)
        for scene_name in args.scenes:
            scene_path = data_root / scene_name
            print(f"\n--- {scene_name} ---")
            if generate_test_txt(scene_path, llffhold=args.llffhold, dry_run=args.dry_run, force=args.force):
                success_count += 1
            else:
                fail_count += 1
    
    print(f"\n--- Summary ---")
    print(f"Processed: {success_count + fail_count}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    exit(main())
