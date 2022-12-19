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
IMAGES_PATH = './THuman/captures_1024_1024/'
SETTINGS_PATH = './THuman/thuman_settings2.json'
NUM_CAMS = 4 # Number of cameras
param = json.load(open(SETTINGS_PATH))
CAM_DATA = [param[cam] for cam in param] # camera data
IMAGES = [cv.imread(os.path.join(IMAGES_PATH,file)) for file in os.listdir(IMAGES_PATH) if file[-4:] == ".png"] # images
DEPTH_MAPS = [np.load(os.path.join(IMAGES_PATH,file)) for file in os.listdir(IMAGES_PATH) if file[-4:] == ".npy"] # depth maps
NUM_IMAGES = len(IMAGES) // NUM_CAMS # number of images
img_index = 0


"""
CREATING CAMERA OBJECTS
"""
cam = []
for i in range(NUM_CAMS):
    cam.append(Camera.Camera(CAM_DATA[i]["img_size"],CAM_DATA[i]["focal_length"],CAM_DATA[i]["img_center"],CAM_DATA[i]["rotation"],CAM_DATA[i]["translation"]))

"""
GENERATING POINT CLOUDS
"""
for img_num in range(NUM_IMAGES):

    for i in range(NUM_CAMS):
        cam[i].add_image(IMAGES[img_index+i],DEPTH_MAPS[img_index+i])
        cam[i].point_cloud()


    print("started timing") # to measure runtime for combiner
    start_time = time.time()
    combiner = Camera.Combiner(cam)
    combiner.combine()
    print("--- %s seconds ---" % (time.time() - start_time))
    np.save("./Point_Clouds/point_cloud_"+str(img_num)+".npy",combiner.complete_pcd)
    combiner.visualize()
    img_index += NUM_CAMS