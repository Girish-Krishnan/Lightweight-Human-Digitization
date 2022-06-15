"""
IMPORTS
"""
from DigitizeLib import Camera
import numpy as np
import open3d as o3d
import time
import json
import os
import cv2 as cv
import pyrealsense2 as rs
import copy

"""
INITIALIZE 2 REALSENSE CAMERAS
"""

pipeline_D435 = rs.pipeline()
config_D435 = rs.config()
config_D435.enable_device('819312073170')
config_D435.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config_D435.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipeline_D435.start(config_D435)

pipeline_D415 = rs.pipeline()
config_D415 = rs.config()
config_D415.enable_device('828612060381')
config_D415.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config_D415.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipeline_D415.start(config_D415)

"""
GET CALIBRATION DATA
"""

SETTINGS_PATH = './data/THuman/calibration_data.json'
param = json.load(open(SETTINGS_PATH))
CAM_DATA = [param[cam] for cam in param] # camera data



"""
TAKE IMAGES USING THE CAMS

"""

try:
    while True:

        # Camera 1
        # Wait for a coherent pair of frames: depth and color
        frames_1 = pipeline_D415.wait_for_frames()
        depth_frame_1 = frames_1.get_depth_frame()
        color_frame_1 = frames_1.get_color_frame()
        if not depth_frame_1 or not color_frame_1:
            continue
        # Convert images to numpy arrays
        depth_image_1 = np.asanyarray(depth_frame_1.get_data())
        color_image_1 = np.asanyarray(color_frame_1.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap_1 = cv.applyColorMap(cv.convertScaleAbs(depth_image_1, alpha=0.5), cv.COLORMAP_JET)

        # Camera 2
        # Wait for a coherent pair of frames: depth and color
        frames_2 = pipeline_D435.wait_for_frames()
        depth_frame_2 = frames_2.get_depth_frame()
        color_frame_2 = frames_2.get_color_frame()
        if not depth_frame_2 or not color_frame_2:
            continue
        # Convert images to numpy arrays
        depth_image_2 = np.asanyarray(depth_frame_2.get_data())
        color_image_2 = np.asanyarray(color_frame_2.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap_2 = cv.applyColorMap(cv.convertScaleAbs(depth_image_2, alpha=0.5), cv.COLORMAP_JET)

        # Stack all images horizontally
        images = np.hstack((color_image_1, depth_colormap_1,color_image_2, depth_colormap_2))

        # Show images from both cameras
        cv.namedWindow('RealSense', cv.WINDOW_NORMAL)
        cv.imshow('RealSense', images)
        cv.waitKey(1)

        # Save images and depth maps from both cameras by pressing 's'
        ch = cv.waitKey(25)
        if ch==115:
            cv.imwrite("my_image_1.jpg",color_image_1)
            np.save("my_depth_1.npy", depth_image_1)
            cv.imwrite("my_image_2.jpg",color_image_2)
            np.save("my_depth_2.npy", depth_image_2)
            print("Save")
            cv.destroyAllWindows()
            break


finally:

    # Stop streaming
    pipeline_D415.stop()
    pipeline_D435.stop()


"""
CREATING CAMERA OBJECTS
"""
cam = []
for i in range(2):
    cam.append(Camera.Camera(CAM_DATA[i]["img_size"],CAM_DATA[i]["focal_length"],CAM_DATA[i]["img_center"],CAM_DATA[i]["rotation"],CAM_DATA[i]["translation"]))


cam[0].add_image(cv.imread("my_image_1.jpg"), np.load("my_depth_1.npy"))
cam[1].add_image(cv.imread("my_image_2.jpg"), np.load("my_depth_2.npy"))

#cam[0].display()
#cam[1].display()

cam[0].point_cloud()
cam[1].point_cloud()

cam[0].visualize()
cam[1].visualize()

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


source = cam[0].pcd_o3d
target = cam[1].pcd_o3d

threshold = 0.02
trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                         [-0.139, 0.967, -0.215, 0.7],
                         [0.487, 0.255, 0.835, -1.4],
                         [0.0, 0.0, 0.0, 1.0]])
draw_registration_result(source, target, trans_init)

print("Initial alignment")
evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
print(evaluation)

print("Apply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
draw_registration_result(source, target, reg_p2p.transformation)

reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 2000))
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
draw_registration_result(source, target, reg_p2p.transformation)

rotation_matrix = np.asarray(reg_p2p.transformation[:3, :3])
translation_vector = np.asarray(reg_p2p.transformation[:3, 3])

cam[1].rotation = np.linalg.inv(rotation_matrix)
cam[1].translation = -translation_vector

combiner = Camera.Combiner(cam)
combiner.combine()
np.save("./Point_Clouds/point_cloud_combined_two_cams.npy",combiner.complete_pcd)
combiner.visualize()