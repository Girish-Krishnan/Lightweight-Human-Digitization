#!/usr/bin/env python
import json
import cv2
import numpy as np
import glob
import pyrealsense2 as rs

pipeline_D415 = rs.pipeline()
config_D415 = rs.config()
config_D415.enable_device('828612060381')
config_D415.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config_D415.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# pipeline_D435 = rs.pipeline()
# config_D435 = rs.config()
# config_D415.enable_device('819312073170')
# config_D415.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
# config_D415.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipeline_D415.start(config_D415)

# Defining the dimensions of checkerboard
CHECKERBOARD = (6,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints_1 = []
objpoints_2 = []

# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints_1 = [] 
imgpoints_2 = []

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Taking pictures and store them in /images/ folder

#setup camera
cap_1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
image_count_1 = 0

cap_2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
image_count_2 = 0


while image_count_1 < 12 and image_count_2 < 12:
    frames_D415 = pipeline_D415.wait_for_frames()
    color_D415 = frames_D415.get_color_frame()
    if not color_D415: continue

    color_img_D415 = np.asanyarray(color_D415.get_data())
    
    cv2.imshow('RealSense_D415', color_img_D415)
    
    c = cv2.waitKey(1)
    if c == 32:
        image_count_1 +=1
        image_count_1 +=1
        cv2.imwrite('images_1/{}.jpg'.format(image_count_1), color_img_D415)
    
pipeline_D415.stop()
cv2.destroyAllWindows()

# Extracting path of individual image stored in a given directory
images_1 = glob.glob('./images_1/*.jpg')

for fname in images_1:
    img_1 = cv2.imread(fname)
    gray_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray_1, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    print(corners)
    print(ret)
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints_1.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray_1, corners, (11,11),(-1,-1), criteria)
        
        imgpoints_1.append(corners2)

        # Draw and display the corners
        img_1 = cv2.drawChessboardCorners(img_1, CHECKERBOARD, corners2, ret)
    
    cv2.imshow('img',img_1)
    cv2.waitKey(0)

cv2.destroyAllWindows()

h,w = img_1.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx_1, dist_1, rvecs_1, tvecs_1 = cv2.calibrateCamera(objpoints_1, imgpoints_1, gray_1.shape[::-1], None, None)

print("Camera matrix : \n")
print(mtx_1)
print("dist : \n")
print(dist_1)
print("rvecs : \n")
print(rvecs_1)
print("tvecs : \n")
print(tvecs_1)
print("Rotation matrix : \n")
print(cv2.Rodrigues(rvecs_1[0])[0])

print("Using cv2.solvePnP : \n")
print(cv2.solvePnP(objpoints_1, imgpoints_1 , mtx_1, dist_1))

data = {
    'mtx_1': mtx_1,
    'dist_1': dist_1,
    'rvecs_1': rvecs_1,
    'tvecs_1': tvecs_1,
    'R' : cv2.Rodrigues(rvecs_1[0])[0]
}

