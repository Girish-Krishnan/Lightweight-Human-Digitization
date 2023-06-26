"""
IMPORTS
"""

from trajectory_utils.trajectory_io import *
import numpy as np
import json
import open3d as o3d
import glob
import matplotlib.pyplot as plt

"""
GET CONFIGS
"""

SETTINGS_PATH = './configuration_parameters.json'
param = json.load(open(SETTINGS_PATH))
cams_list = list(param["cams"].keys())
print("cams_list: ", cams_list)
camera_poses = read_trajectory("odometry.log")

"""
INITIALIZE VOLUME
"""

# TSDF volume
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=2 / 512.0,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

num_images = len(glob.glob(cams_list[0] + "/image_*.jpg"))
print("num_images: ", num_images)

# show RGBD images from all the views
for j in range(num_images):
    for i, cam in enumerate(cams_list):
        # if i == 0: continue
        # print("Integrate {:d}-th image into the volume.".format(i))
        color = o3d.io.read_image(cam + f"/image_{j}.jpg")
        depth = o3d.io.read_image(cam + f"/depth_{j}.png")

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)

        intr = o3d.camera.PinholeCameraIntrinsic(
            width=640,
            height=480,
            fx=param['cams'][cam]['intrinsics']['ir_focal_length'][0],
            fy=param['cams'][cam]['intrinsics']['ir_focal_length'][1],
            cx=param['cams'][cam]['intrinsics']['ir_img_center'][0],
            cy=param['cams'][cam]['intrinsics']['ir_img_center'][1]
        )

        volume.integrate(rgbd, intr, np.linalg.inv(camera_poses[i].pose))

# point cloud generation
pcd = volume.extract_point_cloud()
downpcd = pcd.voxel_down_sample(voxel_size=0.0005)
o3d.visualization.draw_geometries([downpcd])
coordinates = np.asarray(downpcd.points)