from Reconstruction import Camera
import numpy as np
import json
import cv2
import open3d as o3d
import copy
import sys
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import time

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



video_paths = ['./Camera_Data/'+c+'/video.bag' for c in cams_list]
pipelines = []
configs = []
profiles = []
playbacks = []
video_duration = 0
frame_rate = 60

for i in range(len(video_paths)):
    pipelines.append(rs.pipeline())
    configs.append(rs.config())
    rs.config.enable_device_from_file(configs[i], video_paths[i])
    configs[i].enable_stream(rs.stream.depth, rs.format.z16, 60)
    configs[i].enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 60)
    profiles.append(pipelines[i].start(configs[i]))
    playbacks.append(profiles[i].get_device().as_playback())
    # playbacks[i].set_real_time(False)
    video_duration = playbacks[i].get_duration().total_seconds()
    print(video_duration)


num_frames = int(video_duration * frame_rate)

print('Number of frames: ',num_frames)

combiner = Camera.Combiner(cam)



while True:
        for i in range(len(video_paths)):
            # Get frameset of depth
            try:
                frames = pipelines[i].wait_for_frames()

                # Get depth frame
                depth_frame = np.asanyarray(frames.get_depth_frame().get_data())
                
                # Get color frame
                color_frame = np.asanyarray(frames.get_color_frame().get_data())

                cam[i].add_image(color_frame, depth_frame * 0.001)
                cam[i].point_cloud()
            except Exception as e:
                print(e)
                break
            
        combiner.combine()
        pcd_list.append(combiner.pcd_o3d)
        
        o3d.io.write_point_cloud("./reconstructed_video/frame_" + str(len(pcd_list)) + ".ply", combiner.pcd_o3d)

        print(len(pcd_list))
        if len(pcd_list) == num_frames:
            break


"""
VISUALIZATION
"""

"""
vis = o3d.visualization.Visualizer()
vis.create_window()


# geometry is the point cloud used in your animaiton

for i in range(len(pcd_list)):
    # now modify the points of your geometry
    # you can use whatever method suits you best, this is just an example
    geometry = o3d.geometry.PointCloud()
    geometry.points = pcd_list[i].points
    geometry.colors = pcd_list[i].colors
    vis.add_geometry(geometry)
    vis.poll_events()
    vis.update_renderer()
    vis.remove_geometry(geometry)
    time.sleep(1/60)
    #input("Press Enter to continue...")

    """