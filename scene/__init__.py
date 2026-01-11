#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly, fetchPly
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import SH2RGB

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        # Dataset auto-detection
        # - COLMAP: expects images/ and sparse/0/{cameras,images,points3D}.*
        # - ScanNet++: custom nerfstudio + colmap export layout
        # - Blender/NeRF synthetic: transforms_{train,test}.json
        source_path = args.source_path

        is_colmap = os.path.exists(os.path.join(source_path, "sparse"))
        is_blender = os.path.exists(os.path.join(source_path, "transforms_train.json"))
        is_scannetpp = (
            os.path.exists(os.path.join(source_path, "nerfstudio", "transforms_undistorted.json"))
            or os.path.exists(os.path.join(source_path, "nerfstudio", "transforms.json"))
        )

        if is_colmap:
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                source_path,
                args.images,
                args.depths,
                args.eval,
                args.train_test_exp,
            )
        elif is_scannetpp:
            # Local import to avoid circular import with dataset.py during package initialization.
            from dataset import readScannetppInfo

            scene_info = readScannetppInfo(source_path)
        elif is_blender:
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                source_path,
                args.white_background,
                args.depths,
                args.eval,
            )
        else:
            assert False, (
                "Could not recognize scene type. Expected one of: "
                "COLMAP (sparse/), ScanNet++ (nerfstudio/transforms*.json), "
                "or Blender (transforms_train.json)."
            )

        if not self.loaded_iter:
            # with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            #     dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            # Check if we should use random point cloud initialization
            use_random_pcd = getattr(args, 'random_pcd', False)
            if use_random_pcd:
                num_pts = getattr(args, 'random_pcd_num_points', 100000)
                pcd = self._generate_random_point_cloud(scene_info.train_cameras, num_pts)
                print(f"Using random point cloud initialization with {num_pts} points")
            else:
                pcd = scene_info.point_cloud
            self.gaussians.create_from_pcd(pcd, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def _generate_random_point_cloud(self, cam_infos, num_points: int) -> BasicPointCloud:
        """
        Generate a random point cloud based on camera positions.

        Points are distributed in a cube centered at the mean camera position,
        with size based on the camera extent (spread of cameras).

        Args:
            cam_infos: List of camera info objects with R and T attributes
            num_points: Number of random points to generate

        Returns:
            BasicPointCloud with random positions and colors
        """
        # Compute camera centers from extrinsics
        # Camera center in world coords: C = -R^T * T
        cam_centers = []
        for cam in cam_infos:
            R = cam.R  # Already transposed in COLMAP loader
            T = cam.T
            # World position of camera
            center = -R @ T
            cam_centers.append(center)

        cam_centers = np.array(cam_centers)

        # Compute bounding box from camera positions
        center = np.mean(cam_centers, axis=0)
        max_extent = np.max(np.abs(cam_centers - center))

        # Scale extent to cover the scene (cameras typically look inward)
        # Use 2x the camera spread to ensure we cover the scene
        scene_size = max(max_extent * 2.0, 1.0)  # At least 1.0 to avoid degenerate cases

        # Generate random points in a cube centered at the scene center
        xyz = (np.random.random((num_points, 3)) - 0.5) * 2.0 * scene_size + center

        # Random colors (small values in SH space)
        shs = np.random.random((num_points, 3)) / 255.0
        colors = SH2RGB(shs)

        # Zero normals
        normals = np.zeros((num_points, 3))

        return BasicPointCloud(points=xyz, colors=colors, normals=normals)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
