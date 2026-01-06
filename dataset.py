import os
from pathlib import Path
from copy import deepcopy
import json

import numpy as np
from tqdm import tqdm

# from scannetpp_tools.utils.colmap import read_model, qvec2rotmat
from scene.gaussian_model import BasicPointCloud
from scene.colmap_loader import read_extrinsics_text, read_points3D_text, qvec2rotmat
from scene.dataset_readers import CameraInfo, SceneInfo, getNerfppNorm
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from PIL import Image


MAX_NUM_IMAGES_PER_SCENE = 2048

# Fixed seed for any automatically-created train/test split.
# This ensures reproducibility across machines/runs when the dataset does not
# provide an explicit test set.
SPLIT_SEED = 0


def readScannetppInfo(rootdir):
    train_cam_infos = []
    test_cam_infos = []
    transforms_path = os.path.join(rootdir, "nerfstudio/transforms_undistorted.json")
    images_dir = os.path.join(rootdir, "resized_undistorted_images")
    points_txt_path = os.path.join(rootdir, "colmap/points3D.txt")
    camera_extrinsic_path = os.path.join(rootdir, "colmap/images.txt")
    camera_extrinsic = read_extrinsics_text(camera_extrinsic_path)

    extrinsic_dict = {}
    for iamge_id, image in camera_extrinsic.items():
        filename = os.path.basename(image.name)
        R = np.transpose(qvec2rotmat(image.qvec))
        T = np.array(image.tvec)
        extrinsic_dict[filename] = (R, T)
        # Also store a case-insensitive key for robustness on Linux filesystems.
        extrinsic_dict[filename.lower()] = (R, T)

    ply_path = os.path.join(rootdir, "colmap/points3D.ply")

    # Read points3D.txt
    xyz, rgb, _ = read_points3D_text(points_txt_path)
    pcd = BasicPointCloud(points=xyz, colors=rgb / 255.0, normals=np.zeros_like(xyz))

    with open(transforms_path) as f:
        transforms = json.load(f)
    height = transforms["h"]
    width = transforms["w"]
    fx = transforms["fl_x"]
    fy = transforms["fl_y"]

    # Read frames (nerfstudio provides explicit file list; do not rely on directory listing)
    frames = transforms["frames"]
    frames = sorted(frames, key=lambda x: x["file_path"])
    if len(frames) > MAX_NUM_IMAGES_PER_SCENE:
        # Uniformly sample MAX_NUM_IMAGES_PER_SCENE frames
        sample_indices = np.linspace(0, len(frames) - 1, MAX_NUM_IMAGES_PER_SCENE, dtype=np.int32)
        frames = [frames[idx] for idx in sample_indices]

    test_frames = transforms.get("test_frames", None)

    def _frame_has_image_and_extrinsics(frame: dict) -> bool:
        fp = frame.get("file_path", "")
        if not fp:
            return False
        image_path = os.path.join(images_dir, fp)
        if not os.path.exists(image_path):
            return False
        # COLMAP images.txt typically uses the original filename; match case-insensitively too.
        return (fp in extrinsic_dict) or (fp.lower() in extrinsic_dict)

    # Filter out frames that reference missing resized images (common when preprocessing is incomplete)
    frames = [fr for fr in frames if _frame_has_image_and_extrinsics(fr)]
    if test_frames is not None:
        test_frames = [fr for fr in test_frames if _frame_has_image_and_extrinsics(fr)]

    # If no valid test frames remain, create a small test split from training frames.
    # This is required because train_gsplat.py's final evaluation asserts at least 1 test camera.
    if not test_frames or len(test_frames) == 0:
        # Deterministically sample up to 10 test frames from the (filtered) training frames.
        # We use a fixed seed so this split is reproducible.
        n_test = min(10, len(frames))
        if n_test == 0:
            raise FileNotFoundError(
                f"No valid frames found for scene at {rootdir}. "
                f"Expected images under {images_dir} and COLMAP extrinsics under {camera_extrinsic_path}."
            )

        rng = np.random.default_rng(SPLIT_SEED)
        sample_indices = rng.choice(len(frames), size=n_test, replace=False)
        sample_indices = sorted(sample_indices.tolist())
        test_frames = [frames[idx] for idx in sample_indices]
        selected = set(sample_indices)
        frames = [frame for idx, frame in enumerate(frames) if idx not in selected]

    num_train_frames = len(frames)
    
    # Validate dimensions with the first available image
    first_image_path = os.path.join(images_dir, frames[0]["file_path"]) if len(frames) > 0 else os.path.join(images_dir, test_frames[0]["file_path"])
    with Image.open(first_image_path) as img:
        assert img.size[0] == width, f"Image width {img.size[0]} doesn't match transforms width {width}"
        assert img.size[1] == height, f"Image height {img.size[1]} doesn't match transforms height {height}"
    
    for idx, frame in tqdm(enumerate(frames + test_frames), desc="Loading frames", total=len(frames + test_frames)):
        fp = frame["file_path"]
        if fp in extrinsic_dict:
            R, T = extrinsic_dict[fp]
        else:
            R, T = extrinsic_dict[fp.lower()]

        image_path = os.path.join(images_dir, fp)
        image_name = Path(image_path).stem
        FovY = focal2fov(fy, height)
        FovX = focal2fov(fx, width)
        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image_path=image_path,
            image_name=image_name,
            depth_path="",
            depth_params=None,
            width=width,
            height=height,
            is_test=(idx >= num_train_frames),
        )
        if idx < num_train_frames:
            train_cam_infos.append(cam_info)
        else:
            test_cam_infos.append(cam_info)

    # storePly(ply_path, xyz, rgb)
    # pcd = fetchPly(ply_path)
    nerf_normalization = getNerfppNorm(train_cam_infos)
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=False
    )
    return scene_info


if __name__ == "__main__":

    scene_info = readScannetppInfo("/menegroth/scannetpp/data/2024-05-20_17-25/dslr")
