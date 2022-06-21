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
