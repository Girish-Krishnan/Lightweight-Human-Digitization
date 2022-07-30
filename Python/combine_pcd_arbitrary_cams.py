"""
IMPORTS
"""
from Camera import Camera
import numpy as np
import json
import cv2 as cv
import open3d as o3d
import copy
import sys

"""
GET CALIBRATION DATA
"""

SETTINGS_PATH = './configuration_parameters.json'
param = json.load(open(SETTINGS_PATH))
cams_list = list(param["cams"].keys())
CAM_DATA = [param["cams"][cam] for cam in param["cams"]] # camera data

"""
DETERMINING R and T for each cam relative to the first cam
"""

def find_path_to_cam_0(initial_cam):
    unvisited_cams = cams_list.copy()
    min_path = {}
    previous_nodes = {}
    max_value = sys.maxsize

    for cam in unvisited_cams:
        min_path[cam] = max_value

    min_path[cams_list[0]] = 0

    while len(unvisited_cams) > 0:
        current_min_node = None
        for cam in unvisited_cams:
            if current_min_node == None:
                current_min_node = cam
            elif min_path[cam] < min_path[current_min_node]:
                current_min_node = cam


        calibrated_cams = [x for x in list(param["cams"][current_min_node].keys()) if x != "intrinsics"]
        for neighbor in calibrated_cams:
            distance = min_path[current_min_node] + 1
            if distance < min_path[neighbor]:
                min_path[neighbor] = distance
                previous_nodes[neighbor] = current_min_node

        unvisited_cams.remove(current_min_node)
    
    path = []
    node = initial_cam
    while node != cams_list[0]:
        path.append(node)
        node = previous_nodes[node]

    path.append(cams_list[0])

    return path

"""
CREATING CAMERA OBJECTS
"""
cam = []
for i in range(len(cams_list)):
    calibrated_cams = [x for x in list(CAM_DATA[i].keys()) if x != "intrinsics"]
    if len(calibrated_cams) == 0:
        print("No stereocalibration data for camera " + cams_list[i])
        continue

    elif i == 0:
        rotation = [[1.0,0.0,0.0],
                    [0.0,1.0,0.0],
                    [0.0,0.0,1.0]]
        translation = [0.0,0.0,0.0]
        cam.append(Camera.Camera(CAM_DATA[i]["intrinsics"]["img_size"],CAM_DATA[i]["intrinsics"]["focal_length"],CAM_DATA[i]["intrinsics"]["img_center"],rotation,translation))
        path = find_path_to_cam_0(cams_list[i])

    else:
        path = find_path_to_cam_0(cams_list[i])
        rotation = np.eye(3)
        previous_rotation = np.eye(3)
        translation = np.array([0,0,0])
        for j in range(1,len(path)):
            translation = np.add(CAM_DATA[j][path[j]]["translation"],np.matmul(previous_rotation,translation))
            if j != len(path)-1:
                previous_rotation = CAM_DATA[j+1][path[j]]["rotation"]
            rotation = np.matmul(CAM_DATA[j][path[j]]["rotation"],rotation)
        
        cam.append(Camera.Camera(CAM_DATA[i]["intrinsics"]["img_size"],CAM_DATA[i]["intrinsics"]["focal_length"],CAM_DATA[i]["intrinsics"]["img_center"],rotation,translation))
        
for i in range(len(cams_list)):
    cam[i].add_image(cv.imread("./" + cams_list[i] + "/sample_images/image.jpg"), np.load("./" + cams_list[i] + "/sample_images/depth_map.npy"))
    cam[i].point_cloud()
    cam[i].rotate_point_cloud(np.array([[-1,0,0],[0,-1,0],[0,0,1]]))
    cam[i].visualize()

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
