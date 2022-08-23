"""
FINDS THE CORRESPONDING CAMERA PARAMETERS FOR ANY ARBITRARY NUMBER OF CONNECTED CAMERAS

"""

# IMPORTS

import cv2
import numpy as np
import glob
import json
from itertools import combinations
from cv2 import aruco

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
board = aruco.CharucoBoard_create(11, 8, 0.6, 0.45, aruco_dict)


def read_chessboards(images):
    """
    Charuco base pose estimation.
    """
    print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners)>0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize = (3,3),
                                 zeroZone = (-1,-1),
                                 criteria = criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                allCorners.append(res2[1])
                allIds.append(res2[2])

        decimator+=1

    imsize = gray.shape
    return allCorners,allIds,imsize


def calibrate_camera(allCorners,allIds,imsize):
    """
    Calibrates the camera using the dected corners.
    """
    print("CAMERA CALIBRATION")

    cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                 [    0., 1000., imsize[1]/2.],
                                 [    0.,    0.,           1.]])

    distCoeffsInit = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    #flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors


# LOAD INITIAL CONFIG PARAMS, which are then modified in the program

with(open("./configuration_parameters.json")) as f:
    configuration_parameters = json.load(f)
    f.close()

NUM_CALIB_IMGS = configuration_parameters["num_calibration_imgs"]
CHECKERBOARD = (configuration_parameters["checkerboard_dimensions"][0], configuration_parameters["checkerboard_dimensions"][1])
CHECKERBOARD_SIZE = configuration_parameters["checkerboard_size_mm"]  # units: millimeters
IMAGE_TYPE = configuration_parameters["img_file_type"]
NUM_CAMS = len(configuration_parameters["cams"])
THRESHOLD = configuration_parameters["threshold"]

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
serial_numbers = list(configuration_parameters["cams"].keys())

# Defining the world coordinates for 3D points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = CHECKERBOARD_SIZE * objp

# Exception handling when no camera images were found:

if NUM_CAMS == 0:
    print("No camera images found. Please check the directory.")
    exit(-1)

# Extracting path of individual image stored in a given directory

images = []
for i in range(NUM_CAMS):
    images.append(glob.glob("./" + serial_numbers[i] + "/calibration_images" + "/*" + IMAGE_TYPE))

# Exception handling when only one camera was detected - only intrinsic calibration is done.

if NUM_CAMS == 1:
    print("Only one camera detected. Performing only intrinsic cailbration.")
    images = images[0]
    
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []

    # Creating vector to store vectors of 2D points for each checkerboard image

    imgpoints = []

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        #print(corners)
        #print(ret)
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        
        cv2.imshow('img',img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    h,w = img.shape[:2]

    """
    Performing intrinsic camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)

    configuration_parameters["cams"][serial_numbers[0]]["intrinsics"] = {}
    configuration_parameters["cams"][serial_numbers[0]]["intrinsics"]["img_size"] = [gray.shape[::-1][0], gray.shape[::-1][1]]
    configuration_parameters["cams"][serial_numbers[0]]["intrinsics"]["focal_length"] = [mtx[0][0], mtx[1][1]]
    configuration_parameters["cams"][serial_numbers[0]]["intrinsics"]["img_center"] = [mtx[0][2], mtx[1][2]]
    json.dump(configuration_parameters, open("configuration_parameters.json", "w"), indent = 4)
    exit(-1)

image_pairs = combinations(range(NUM_CAMS),2) # finding all distinct pairs of cameras

for pair in image_pairs:
    x = pair[0]
    y = pair[1]
    success_count = 0 # the number of common images between the two cameras that contain the chessboard successfully detected
    
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []

    # Creating vector to store vectors of 2D points for each checkerboard image

    imgpoints_1 = []
    imgpoints_2 = []

    img_set_1 = images[x]
    img_set_2 = images[y]

    allCorners_1, allIds_1, imsize_1 = read_chessboards(img_set_1)
    allCorners_2, allIds_2, imsize_2 = read_chessboards(img_set_2)

    print(allCorners_1)
    print(allIds_1)
    

    ret_1, mtx_1, dist_1, rvecs_1, tvecs_1 = calibrate_camera(allCorners_1, allIds_1, imsize_1)
    ret_2, mtx_2, dist_2, rvecs_2, tvecs_2 = calibrate_camera(allCorners_2, allIds_2, imsize_2)

    # for i in range(NUM_CALIB_IMGS):
        # img_1 = cv2.imread(images[x][i])
        # img_2 = cv2.imread(images[y][i])
        #
        # gray_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
        # gray_2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)
        #
        # # Find the chess board corners
        # # If desired number of corners are found in the image then ret = true
        # ret_1, corners_1 = cv2.findChessboardCorners(gray_1, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        # ret_2, corners_2 = cv2.findChessboardCorners(gray_2, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        # print("Cam 1: \n")
        # print(corners_1)
        # print(ret_1)

        # print("Cam 2: \n")
        # print(corners_2)
        # print(ret_2)
        #
        # """
        # If desired number of corner are detected,
        # we refine the pixel coordinates and display
        # them on the images of checker board
        # """
        #
        # if ret_1 == True and ret_2 == True:
        #     success_count += 1
        #     objpoints.append(objp)
        #
        #     # refining pixel coordinates for given 2d points.
        #     corners_1 = cv2.cornerSubPix(gray_1, corners_1, (11,11),(-1,-1), criteria)
        #     corners_2 = cv2.cornerSubPix(gray_2, corners_2, (11,11),(-1,-1), criteria)
        #
        #     imgpoints_1.append(corners_1)
        #     imgpoints_2.append(corners_2)
        #
        #     # Draw and display the corners
        #     img_1 = cv2.drawChessboardCorners(img_1, CHECKERBOARD, corners_1, ret_1)
        #     img_2 = cv2.drawChessboardCorners(img_2, CHECKERBOARD, corners_2, ret_2)
        #     images_display = np.hstack((img_1, img_2))
        #
        #     cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        #     cv2.imshow('RealSense', images_display)
        #     cv2.waitKey(0)


    # cv2.destroyAllWindows()

    # if success_count < THRESHOLD:
    #     print("Cams " + str(x) + " and " + str(y) + " do not have enough common images. Skipping calibration.")
    #     continue
    #
    # """
    # Performing camera calibration by
    # passing the value of known 3D points (objpoints)
    # and corresponding pixel coordinates of the
    # detected corners (imgpoints)
    # """
    # ret_1, mtx_1, dist_1, rvecs_1, tvecs_1 = cv2.calibrateCamera(objpoints, imgpoints_1, gray_1.shape[::-1], None, None)
    # height_1, width_1, channels_1 = img_1.shape
    # mtx_1, roi_1 = cv2.getOptimalNewCameraMatrix(mtx_1, dist_1, (width_1, height_1), 1, (width_1, height_1))
    #
    # ret_2, mtx_2, dist_2, rvecs_2, tvecs_2 = cv2.calibrateCamera(objpoints, imgpoints_2, gray_2.shape[::-1], None, None)
    # height_2, width_2, channels_2 = img_2.shape
    # mtx_2, roi_2 = cv2.getOptimalNewCameraMatrix(mtx_2, dist_2, (width_2, height_2), 1, (width_2, height_2))

    flags = cv2.CALIB_FIX_INTRINSIC

    criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_1, imgpoints_2, mtx_1, dist_1, mtx_2, dist_2, [640, 480], criteria_stereo, flags)

    print("Stereo Calibration RMS for cam " + str(x) + " and cam " + str(y) + ": ", rms)

    mtx_1 = K1.tolist()
    mtx_2 = K2.tolist()
    T = T.tolist()
    T = [T[0][0],T[1][0],T[2][0]]

    configuration_parameters["cams"][serial_numbers[x]]["intrinsics"] = {}
    configuration_parameters["cams"][serial_numbers[x]][serial_numbers[y]] = {}
    configuration_parameters["cams"][serial_numbers[x]]["intrinsics"]["img_size"] = [640, 480]
    configuration_parameters["cams"][serial_numbers[x]]["intrinsics"]["focal_length"] = [mtx_1[0][0], mtx_1[1][1]]
    configuration_parameters["cams"][serial_numbers[x]]["intrinsics"]["img_center"] = [mtx_1[0][2], mtx_1[1][2]]
    configuration_parameters["cams"][serial_numbers[x]][serial_numbers[y]]["rotation"] = np.eye(3).tolist()
    configuration_parameters["cams"][serial_numbers[x]][serial_numbers[y]]["translation"] =  [0,0,0]

    configuration_parameters["cams"][serial_numbers[y]]["intrinsics"] = {}
    configuration_parameters["cams"][serial_numbers[y]][serial_numbers[x]] = {}
    configuration_parameters["cams"][serial_numbers[y]]["intrinsics"]["img_size"] = [640, 480]
    configuration_parameters["cams"][serial_numbers[y]]["intrinsics"]["focal_length"] = [mtx_2[0][0], mtx_2[1][1]]
    configuration_parameters["cams"][serial_numbers[y]]["intrinsics"]["img_center"] = [mtx_2[0][2], mtx_2[1][2]]
    configuration_parameters["cams"][serial_numbers[y]][serial_numbers[x]]["rotation"] = R.tolist()
    configuration_parameters["cams"][serial_numbers[y]][serial_numbers[x]]["translation"] = T

    json.dump(configuration_parameters, open("configuration_parameters.json", "w"), indent = 4)