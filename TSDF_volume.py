"""
IMPORTS
"""
from Reconstruction import Camera
import numpy as np
import json
import cv2 as cv
import open3d as o3d
import sys 
import glob

"""
GET CONFIGS
"""

SETTINGS_PATH = './configuration_parameters.json'
param = json.load(open(SETTINGS_PATH))
cams_list = list(param["cams"].keys())
print("cams_list: ", cams_list)
CAM_DATA = [param["cams"][cam] for cam in param["cams"]]  # camera data

"""
INITIALIZE VOLUME
"""

volume = o3d.geometry.TSDFVolume(
    voxel_length=0.01,  # Size of each voxel in meters
    sdf_trunc=0.03  # Truncation distance in meters
)

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

    if i == 0:
        rotation = [[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]]
        translation = [0.0, 0.0, 0.0]
        cam.append(Camera.Camera(CAM_DATA[i]["intrinsics"]["img_size"], CAM_DATA[i]["intrinsics"]["ir_focal_length"],
                                 CAM_DATA[i]["intrinsics"]["ir_img_center"], rotation, translation))
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
        print("Current Cam: ", cams_list[i])
        print("path to cam0: ", path)
        print("Final rotation: \n", rotation)
        print("Final translation: ", translation)
        print("___")
        cam.append(Camera.Camera(CAM_DATA[i]["intrinsics"]["img_size"], CAM_DATA[i]["intrinsics"]["ir_focal_length"],
                                 CAM_DATA[i]["intrinsics"]["ir_img_center"], rotation, translation))


directory = "./static/"
num_images = len(glob.glob(directory + cams_list[0] + "/*.jpg"))

for j in range(num_images):
    for i in range(len(cams_list)):
        cam[i].add_image(cv.imread("./static/" + cams_list[i] + f"/image_{j}.jpg"),
                        np.load("./static/" + cams_list[i] + f"/depth_map_{j}.npy") * 0.001)
        cam[i].point_cloud()
        volume.integrate(cam[i].pcd_o3d)

# Extract mesh
mesh = volume.extract_triangle_mesh()
o3d.visualization.draw_geometries([mesh])  # Visualize the mesh
o3d.io.write_triangle_mesh("reconstructed_mesh.ply", mesh)  # Save the mesh to a file
