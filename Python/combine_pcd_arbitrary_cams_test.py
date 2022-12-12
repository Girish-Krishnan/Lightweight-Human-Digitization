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

SETTINGS_PATH = './data/THuman/thuman_settings2.json'
param = json.load(open(SETTINGS_PATH))
cams_list = list(param.keys())
CAM_DATA = [param[cam] for cam in param]  # camera data


"""
CREATING CAMERA OBJECTS
"""
cam = []
for i in range(len(cams_list)):
    cam.append(Camera.Camera(CAM_DATA[i]["img_size"], CAM_DATA[i]["focal_length"],
                             CAM_DATA[i]["img_center"], CAM_DATA[i]['rotation'], CAM_DATA[i]['translation']))

for i in range(len(cams_list)):
    cam[i].add_image(cv.imread("./data/THuman/captures_1024_1024/0050_0{}.png".format(i)),
                     np.load("./data/THuman/captures_1024_1024/0050_0{}.npy".format(i)))
    cam[i].point_cloud()
    cam[i].visualize()

combiner = Camera.Combiner(cam)
combiner.combine()
combiner.visualize()
