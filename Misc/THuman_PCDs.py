"""
IMPORTS
"""
from Camera import Camera
import numpy as np
import time
import json
import os
import cv2 as cv

"""
CONSTANTS + EXTRACTED DATA
"""
IMAGE_DIR = './THuman/captures_1024_1024/'
SETTINGS_FILE = './THuman/thuman_settings2.json'
NUM_CAMS = 4 # Number of cameras
settings = json.load(open(SETTINGS_FILE))
camera_data = [settings[cam] for cam in settings] # camera data
images = [cv.imread(os.path.join(IMAGE_DIR, file)) for file in os.listdir(IMAGE_DIR) if file.endswith('.png')]
depth_maps = [np.load(os.path.join(IMAGE_DIR, file)) for file in os.listdir(IMAGE_DIR) if file.endswith('.npy')]
num_images = len(images) // NUM_CAMS # number of images
img_index = 0


"""
CREATING CAMERA OBJECTS
"""

cams = [Camera.Camera(camera_data[i]["img_size"],
camera_data[i]["focal_length"],
camera_data[i]["img_center"],
camera_data[i]["rotation"],
camera_data[i]["translation"]) for i in range(NUM_CAMS)]

"""
GENERATING POINT CLOUDS
"""

for img_num in range(num_images):
    for i in range(NUM_CAMS):
        # Add image and depth map to camera object, generate point cloud and visualize it
        cams[i].add_image(images[img_index + i], depth_maps[img_index + i])
        cams[i].point_cloud()
        cams[i].visualize()

        print("Started timing")  # To measure runtime for combiner
        start_time = time.time()
        combiner = Camera.Combiner(cams)
        combiner.combine()
        print("--- %s seconds ---" % (time.time() - start_time))

        # Save point cloud and visualize it
        np.save(f"./Point_Clouds/point_cloud_{img_num}.npy", combiner.complete_pcd)
        combiner.visualize()

        img_index += NUM_CAMS  # Move to next set of images

