"""
READS IMAGES IN calibration_images/ DIRECTORY

AND FINDS THE CORRESPONDING CAMERA PARAMETERS

"""

import cv2
import numpy as np
import glob
import json

# Defining the dimensions of checkerboard
CHECKERBOARD = (6,9)
CHECKERBOARD_SIZE = 20  # units: millimeters
IMAGE_TYPE = ".jpg"
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image

objpoints = []

# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints_1 = [] # for D415
imgpoints_2 = [] # for D435

# Defining the world coordinates for 3D points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = CHECKERBOARD_SIZE * objp
#print(objp)

# Extracting path of individual image stored in a given directory

images_1 = sorted(glob.glob('./calibration_images/D415/*' + IMAGE_TYPE))  # images from D415
images_2 = sorted(glob.glob('./calibration_images/D435/*' + IMAGE_TYPE))  # images from D435

for i in range(len(images_1)):
    img_1 = cv2.imread(images_1[i])
    img_2 = cv2.imread(images_2[i])

    gray_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret_1, corners_1 = cv2.findChessboardCorners(gray_1, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret_2, corners_2 = cv2.findChessboardCorners(gray_2, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # print("D415: \n")
    # print(corners_1)
    # print(ret_1)

    # print("D435: \n")
    # print(corners_2)
    # print(ret_2)

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """

    if ret_1 == True and ret_2 == True:
        objpoints.append(objp)

        # refining pixel coordinates for given 2d points.
        corners_1 = cv2.cornerSubPix(gray_1, corners_1, (11,11),(-1,-1), criteria)
        corners_2 = cv2.cornerSubPix(gray_2, corners_2, (11,11),(-1,-1), criteria)
        
        imgpoints_1.append(corners_1)
        imgpoints_2.append(corners_2)

        # Draw and display the corners
        # img_1 = cv2.drawChessboardCorners(img_1, CHECKERBOARD, corners_1, ret_1)
        # img_2 = cv2.drawChessboardCorners(img_2, CHECKERBOARD, corners_2, ret_2)
        # images = np.hstack((img_1, img_2))
    
        # cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        # cv2.imshow('RealSense', images)
        # cv2.waitKey(0)


cv2.destroyAllWindows()


"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret_1, mtx_1, dist_1, rvecs_1, tvecs_1 = cv2.calibrateCamera(objpoints, imgpoints_1, gray_1.shape[::-1], None, None)
height_1, width_1, channels_1 = img_1.shape
mtx_1, roi_1 = cv2.getOptimalNewCameraMatrix(mtx_1, dist_1, (width_1, height_1), 1, (width_1, height_1))

ret_2, mtx_2, dist_2, rvecs_2, tvecs_2 = cv2.calibrateCamera(objpoints, imgpoints_2, gray_2.shape[::-1], None, None)
height_2, width_2, channels_2 = img_2.shape
mtx_2, roi_2 = cv2.getOptimalNewCameraMatrix(mtx_2, dist_2, (width_2, height_2), 1, (width_2, height_2))

flags = cv2.CALIB_FIX_INTRINSIC

criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_1, imgpoints_2, mtx_1, dist_1, mtx_2, dist_2, gray_1.shape[::-1], criteria_stereo, flags)

print("Stereo Calibration RMS: ", rms)

with open("data/THuman/calibration_data.json", 'r') as f:
    data = json.load(f)
    mtx_1 = K1.tolist()
    mtx_2 = K2.tolist()
    T = T.tolist()
    T = [T[0][0],T[1][0],T[2][0]]

    data["cam_D415"]["img_size"] = [gray_1.shape[::-1][0], gray_1.shape[::-1][1]]
    data["cam_D415"]["focal_length"] = [mtx_1[0][0], mtx_1[1][1]]
    data["cam_D415"]["img_center"] = [mtx_1[0][2], mtx_1[1][2]]
    data["cam_D415"]["rotation"] = np.eye(3).tolist()
    data["cam_D415"]["translation"] =  [0,0,0]

    data["cam_D435"]["img_size"] = [gray_2.shape[::-1][0], gray_2.shape[::-1][1]]
    data["cam_D435"]["focal_length"] = [mtx_2[0][0], mtx_2[1][1]]
    data["cam_D435"]["img_center"] = [mtx_2[0][2], mtx_2[1][2]]
    data["cam_D435"]["rotation"] = R.tolist()
    data["cam_D435"]["translation"] = T

    json.dump(data, open("data/THuman/calibration_data.json", "w"), indent = 4)

with open("data/THuman/params.json", 'r') as f:
    data = json.load(f)

    data["cam_D435"]["img_size"] = [gray_2.shape[::-1][0], gray_2.shape[::-1][1]]
    data["cam_D435"]["rotation"] = R.tolist()
    data["cam_D435"]["translation"] = T

    json.dump(data, open("data/THuman/params.json", "w"), indent = 4)


