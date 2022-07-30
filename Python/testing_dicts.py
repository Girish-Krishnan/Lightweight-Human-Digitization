from Camera import Camera
import numpy as np
import json
import cv2 as cv
import open3d as o3d
import copy

"""
GET CALIBRATION DATA
"""

SETTINGS_PATH = './configuration_parameters.json'
param = json.load(open(SETTINGS_PATH))
cams_list = list(param["cams"].keys())
cams_list.append("camera_0")
CAM_DATA = [param["cams"][cam] for cam in param["cams"]] # camera data

"""
CREATING CAMERA OBJECTS
"""
cam = []
for i in range(len(cams_list)):
    cam.append(Camera.Camera(CAM_DATA[i]["intrinsics"]["img_size"],CAM_DATA[i]["intrinsics"]["focal_length"],CAM_DATA[i]["intrinsics"]["img_center"],CAM_DATA[i][cams_list[2]]["rotation"],CAM_DATA[i][cams_list[2]]["translation"]))