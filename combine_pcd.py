"""
IMPORTS
"""
from Reconstruction import Camera
import numpy as np
import json
import cv2 as cv
import open3d as o3d
import sys
import argparse
from trajectory_utils.trajectory_io import *

RECONST_IMAGES_DIR = '/reconstruction_images'
CALIB_IMAGES_DIR = '/calibration_images'

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='./configuration_parameters.json')
parser.add_argument('--output_file', type=str, default='./point_cloud_combined.ply')
parser.add_argument('--data_dir', type=str, default='./Capture_Data')
parser.add_argument('--save_individual', action='store_true')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--odom_file', type=str, default='./odometry.log')
parser.add_argument('--mesh_file', type=str, default='./mesh_combined.ply')
args = parser.parse_args()

"""
GET CALIBRATION DATA
"""

SETTINGS_PATH = args.config_file
param = json.load(open(SETTINGS_PATH))
cams_list = list(param["cams"].keys())
#print("cams_list: ", cams_list)
CAM_DATA = [param["cams"][cam] for cam in param["cams"]]  # camera data
# Empty the odom file
with(open(args.odom_file, "w")) as f:
    f.close()

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
        #print("No stereocalibration data for camera " + cams_list[i])
        continue

    if i == 0:
        rotation = [[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]]
        translation = [0.0, 0.0, 0.0]
        cam.append(Camera.Camera(CAM_DATA[i]["intrinsics"]["img_size"], CAM_DATA[i]["intrinsics"]["ir_focal_length"],
                                 CAM_DATA[i]["intrinsics"]["ir_img_center"], rotation, translation, cams_list[i], args.data_dir))
        path = find_path_to_cam_0(cams_list[i])


    else:
        path = find_path_to_cam_0(cams_list[i])
        rotation = np.eye(3)
        previous_rotation = np.eye(3)
        translation = np.array([0, 0, 0])
        for j in range(1, len(path)):
            idx = cams_list.index(path[j - 1])
            previous_rotation = CAM_DATA[idx][path[j]]["rotation"]
            translation = np.add(CAM_DATA[idx][path[j]]["translation"], np.matmul(previous_rotation, translation))
            rotation = np.matmul(previous_rotation, rotation)
        #print("Current Cam: ", cams_list[i])
        #print("path to cam0: ", path)
        #print("Final rotation: \n", rotation)
        #print("Final translation: ", translation)
        #print("___")
        cam.append(Camera.Camera(CAM_DATA[i]["intrinsics"]["img_size"], CAM_DATA[i]["intrinsics"]["ir_focal_length"],
                                 CAM_DATA[i]["intrinsics"]["ir_img_center"], rotation, translation, cams_list[i], args.data_dir))

    trans = np.vstack( (np.hstack((np.array(rotation), np.array(translation).reshape(-1,1))), [0,0,0,1]) )
    r_mat = trans[0:3, 0:3]
    t = trans[0:3, 3] 
    write_to_file(args.odom_file, 0, np.eye(3), [0, 0, 0])
    write_to_file(args.odom_file, 1, r_mat, t)

for i in range(len(cams_list)):
    cam[i].add_image(cv.imread(args.data_dir + "/" + cams_list[i] + RECONST_IMAGES_DIR + "/image.jpg"),
                     np.load(args.data_dir + "/" + cams_list[i] + RECONST_IMAGES_DIR + "/depth_map.npy") * 0.001)
    cam[i].point_cloud()
    if args.save_individual:
        o3d.io.write_point_cloud(args.data_dir + "/" + cams_list[i] + '/individual_pcd' + ".ply", cam[i].pcd_o3d)

combiner = Camera.Combiner(cam)
combiner.combine()
o3d.io.write_point_cloud(args.output_file, combiner.pcd_o3d)
#o3d.io.write_triangle_mesh(args.mesh_file, combiner.mesh_o3d)
o3d.io.write_triangle_mesh(args.mesh_file.split('.')[0] + '-computed.ply', combiner.mesh_o3d_poisson)

# if args.visualize:
#     combiner.visualize()