import numpy as np
import json
import cv2
import pyrealsense2 as rs

"""
GET CAM CALIBRATION DATA
"""

SETTINGS_PATH = './configuration_parameters.json'
param = json.load(open(SETTINGS_PATH))
cams_list = list(param["cams"].keys())
print("cams_list: ", cams_list)

for c in cams_list:

    # Load the depth map
    depth_map = np.load("./" + c + "/sample_images/depth_map.npy")

    # Apply morphological closing operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filtered_depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, kernel)
    for j in range(100):
        filtered_depth_map = cv2.morphologyEx(filtered_depth_map, cv2.MORPH_CLOSE, kernel)

    # Save the filtered depth map
    cv2.imwrite("./" + c + "/sample_images/filtered_depth_map.png", cv2.applyColorMap(cv2.convertScaleAbs(filtered_depth_map, alpha=0.03),cv2.COLORMAP_JET))
    np.save("./" + c + "/sample_images/filtered_depth_map.npy",filtered_depth_map)
