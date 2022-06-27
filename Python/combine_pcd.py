"""
IMPORTS
"""
from Camera import Camera
import numpy as np
import json
import cv2 as cv
import open3d as o3d
import copy

"""
GET CALIBRATION DATA
"""

SETTINGS_PATH = './data/THuman/calibration_data.json'
param = json.load(open(SETTINGS_PATH))
CAM_DATA = [param[cam] for cam in param] # camera data

"""
CREATING CAMERA OBJECTS
"""
cam = []
for i in range(2):
    cam.append(Camera.Camera(CAM_DATA[i]["img_size"],CAM_DATA[i]["focal_length"],CAM_DATA[i]["img_center"],CAM_DATA[i]["rotation"],CAM_DATA[i]["translation"]))


cam[0].add_image(cv.imread("./sample_images/D415_sample.jpg"), np.load("./sample_images/D415_sample.npy"))
cam[1].add_image(cv.imread("./sample_images/D435_sample.jpg"), np.load("./sample_images/D435_sample.npy"))

cam[0].point_cloud()
cam[1].point_cloud()

cam[0].rotate_point_cloud(np.array([[-1,0,0],[0,-1,0],[0,0,1]]))
cam[1].rotate_point_cloud(np.array([[-1,0,0],[0,-1,0],[0,0,1]]))

# cam[0].visualize()
# cam[1].visualize()

# def draw_registration_result(source, target, transformation):
#     source_temp = copy.deepcopy(source)
#     target_temp = copy.deepcopy(target)
#     source_temp.paint_uniform_color([1, 0.706, 0])
#     target_temp.paint_uniform_color([0, 0.651, 0.929])
#     source_temp.transform(transformation)
#     o3d.visualization.draw_geometries([source_temp, target_temp],
#                                       zoom=0.4459,
#                                       front=[0.9288, -0.2951, -0.2242],
#                                       lookat=[1.6784, 2.0612, 1.4451],
#                                       up=[-0.3402, -0.9189, -0.1996])


# source = cam[0].pcd_o3d
# target = cam[1].pcd_o3d

# threshold = 0.02
# trans_init = np.asarray(cam[1].extrinsic_matrix)
# draw_registration_result(source, target, trans_init)

# print("Initial alignment")
# evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
# print(evaluation)

# print("Apply point-to-point ICP")
# reg_p2p = o3d.pipelines.registration.registration_icp(
#         source, target, threshold, trans_init,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint())
# print(reg_p2p)
# print("Transformation is:")
# print(reg_p2p.transformation)
# draw_registration_result(source, target, reg_p2p.transformation)

# reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 2000))
# print(reg_p2p)
# print("Transformation is:")
# print(reg_p2p.transformation)
# draw_registration_result(source, target, reg_p2p.transformation)


combiner = Camera.Combiner(cam)
combiner.combine()
np.save("./data/Point_Clouds/point_cloud_combined.npy",combiner.complete_pcd)
combiner.visualize()

# viewer = o3d.visualization.Visualizer()
# viewer.create_window()
# viewer.add_geometry(combiner.pcd_o3d)
# opt = viewer.get_render_option()
# opt.show_coordinate_frame = True
# opt.background_color = np.asarray([0.5, 0.5, 0.5])
# viewer.run()
# viewer.destroy_window()
