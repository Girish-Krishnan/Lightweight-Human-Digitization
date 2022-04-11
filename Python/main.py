"""
IMPORTS
"""
from DigitizeLib import Camera
import numpy as np
import open3d as o3d
import time
import json
import os
import cv2 as cv

"""
CONSTANTS + EXTRACTED DATA
"""

NUM_CAMS = 4 # Number of cameras
param = json.load(open('./data/THuman/thuman_settings2.json'))
CAM_DATA = [param["cam_"+str(i)+"_r"] for i in range(NUM_CAMS)] # camera data
IMAGES = [cv.imread(os.path.join('./data/THuman/captures_1024_1024/',file)) for file in os.listdir("./data/THuman/captures_1024_1024/") if file[-4:] == ".png"] # images
DEPTH_MAPS = [np.load(os.path.join('./data/THuman/captures_1024_1024/',file)) for file in os.listdir("./data/THuman/captures_1024_1024/") if file[-4:] == ".npy"] # depth maps
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
    np.save("./Point_Clouds/point_cloud_"+str(img_num)+".npy",combiner.complete_pcd)
    print("--- %s seconds ---" % (time.time() - start_time))

    combiner.visualize()
    img_index += NUM_CAMS