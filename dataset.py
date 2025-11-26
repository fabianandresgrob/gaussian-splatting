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

    # Read frames
    frames = transforms["frames"]
    # Sort frames by file_path
    frames = sorted(frames, key=lambda x: x["file_path"])
    if len(frames) > MAX_NUM_IMAGES_PER_SCENE:
        # Uniformly sample MAX_NUM_IMAGES_PER_SCENE frames
        sample_indices = np.linspace(0, len(frames) - 1, MAX_NUM_IMAGES_PER_SCENE, dtype=np.int32)
        frames = [frames[idx] for idx in sample_indices]

    if "test_frames" in transforms:
        test_frames = transforms["test_frames"]
    else:
        # Subsample 10 test frames from the training frames
        sample_indices = np.linspace(0, len(frames) - 1, 10, dtype=np.int32)
        test_frames = [frames[idx] for idx in sample_indices]
        frames = [frame for idx, frame in enumerate(frames) if idx not in sample_indices]

    num_train_frames = len(frames)
    
    # Validate dimensions with the first image
    first_image_path = os.path.join(images_dir, frames[0]["file_path"])
    with Image.open(first_image_path) as img:
        assert img.size[0] == width, f"Image width {img.size[0]} doesn't match transforms width {width}"
        assert img.size[1] == height, f"Image height {img.size[1]} doesn't match transforms height {height}"
    
    for idx, frame in tqdm(enumerate(frames + test_frames), desc="Loading frames", total=len(frames + test_frames)):
        R, T = extrinsic_dict[frame["file_path"]]

        image_path = os.path.join(images_dir, frame["file_path"])
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
