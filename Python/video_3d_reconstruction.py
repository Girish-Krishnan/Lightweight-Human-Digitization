from Camera import Camera
import numpy as np
import json
import cv2
import open3d as o3d
import copy
import sys
import matplotlib.pyplot as plt
import pyrealsense2 as rs

"""
GET CAM CALIBRATION DATA
"""

SETTINGS_PATH = './configuration_parameters.json'
param = json.load(open(SETTINGS_PATH))
cams_list = list(param["cams"].keys())
print("cams_list: ", cams_list)
CAM_DATA = [param["cams"][cam] for cam in param["cams"]]  # camera data
pcd_list = [] # stores the pcds in each frame

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



video_paths = ['./'+c+'/video.bag' for c in cams_list]
pipelines = []
configs = []

for i in range(len(video_paths)):
    pipelines.append(rs.pipeline())
    configs.append(rs.config())
    rs.config.enable_device_from_file(configs[i], video_paths[i])
    configs[i].enable_stream(rs.stream.depth, rs.format.z16, 30)
    pipelines[i].start(configs[i])

combiner = Camera.Combiner(cam)

try:
    while True:
        for i in range(len(video_paths)):
            # Get frameset of depth
            frames = pipelines[i].wait_for_frames()

            # Get depth frame
            depth_frame = np.asanyarray(frames.get_depth_frame().get_data())
            
            # Get color frame
            color_frame = np.asanyarray(frames.get_color_frame().get_data())

            # Colorize depth frame to jet colormap
            #depth_color_frame = colorizer.colorize(depth_frame)

            # Convert depth_frame to numpy array to render image in opencv
            #depth_color_image = np.asanyarray(depth_color_frame.get_data())

            # # Render image in opencv window
            #cv2.imshow("Depth Stream", depth_color_image)
            #key = cv2.waitKey(1)
            # # if pressed escape exit program
            #if key == 27:
            #    cv2.destroyAllWindows()
            #    break

            cam[i].add_image(color_frame, depth_frame * 0.001)
            cam[i].point_cloud()
            
        combiner.combine()
        pcd_list.append(combiner.pcd_o3d)


finally:
    pass


"""
VISUALIZATION
"""


vis = o3d.visualization.Visualizer()
vis.create_window()

# geometry is the point cloud used in your animaiton
geometry = o3d.geometry.PointCloud()
vis.add_geometry(geometry)

for i in range(pcd_list):
    # now modify the points of your geometry
    # you can use whatever method suits you best, this is just an example
    geometry.points = pcd_list[i].points
    vis.update_geometry(geometry)
    vis.poll_events()
    vis.update_renderer()