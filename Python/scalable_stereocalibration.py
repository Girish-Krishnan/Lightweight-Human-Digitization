"""
FINDS THE CORRESPONDING CAMERA PARAMETERS FOR ANY ARBITRARY NUMBER OF CONNECTED CAMERAS

"""

# IMPORTS

import cv2
import numpy as np
import glob
import json
from itertools import combinations
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation
from utils.trajectory_io import *

# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_250)
CHARUCOBOARD_ROWCOUNT = 9
CHARUCOBOARD_COLCOUNT = 12

distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Create grid board object we're using in our stream
CHARUCO_BOARD = aruco.CharucoBoard_create(
    squaresX=CHARUCOBOARD_COLCOUNT,
    squaresY=CHARUCOBOARD_ROWCOUNT,
    squareLength=0.060,
    markerLength=0.044,
    dictionary=ARUCO_DICT)


# LOAD INITIAL CONFIG PARAMS, which are then modified in the program

with(open("./configuration_parameters.json")) as f:
    configuration_parameters = json.load(f)
    f.close()

NUM_CALIB_IMGS = configuration_parameters["num_calibration_imgs"]
CHECKERBOARD = (
configuration_parameters["checkerboard_dimensions"][0], configuration_parameters["checkerboard_dimensions"][1])
CHECKERBOARD_SIZE = configuration_parameters["checkerboard_size_mm"]  # units: millimeters
CHECKERBOARD_SIZE *= 0.001
IMAGE_TYPE = configuration_parameters["img_file_type"]
NUM_CAMS = len(configuration_parameters["cams"])
THRESHOLD = configuration_parameters["threshold"]

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
serial_numbers = list(configuration_parameters["cams"].keys())

# Defining th   for 3D points: order is important!!!
objp = np.zeros((CHECKERBOARD[1] * CHECKERBOARD[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[1], 0:CHECKERBOARD[0]].T.reshape(-1, 2)
objp = CHECKERBOARD_SIZE * objp
# print(objp)

# Exception handling when no camera images were found:

if NUM_CAMS == 0:
    print("No camera images found. Please check the directory.")
    exit(-1)

# Extracting path of individual image stored in a given directory
images = []
for i in range(NUM_CAMS):
    images.append(glob.glob("./" + serial_numbers[i] + "/calibration_images" + "/*" + IMAGE_TYPE))

image_pairs = combinations(range(NUM_CAMS), 2)  # finding all distinct pairs of cameras

for pair in image_pairs:
    x = pair[0]
    y = pair[1]
    common_img_count = 0  # the number of common images between the two cameras that contain the chessboard successfully detected

    cam1_f = configuration_parameters["cams"][serial_numbers[x]]["intrinsics"]["ir_focal_length"]
    cam1_c = configuration_parameters["cams"][serial_numbers[x]]["intrinsics"]["ir_img_center"]
    cam1_mtx = np.array([
        [cam1_f[0], 0, cam1_c[0]],
        [0, cam1_f[1], cam1_c[1]],
        [0, 0, 1]
    ])
    cam2_f = configuration_parameters["cams"][serial_numbers[y]]["intrinsics"]["ir_focal_length"]
    cam2_c = configuration_parameters["cams"][serial_numbers[y]]["intrinsics"]["ir_img_center"]
    cam2_mtx = np.array([
        [cam2_f[0], 0, cam2_c[0]],
        [0, cam2_f[1], cam2_c[1]],
        [0, 0, 1]
    ])

    objpoints = []  # Creating vector to store vectors of 3D points for each checkerboard image
    imgpoints_1 = []  # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints_2 = []

    for i in range(NUM_CALIB_IMGS):
        img_1 = cv2.imread(images[x][i])
        img_2 = cv2.imread(images[y][i])

        gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

        # Find the aruco board corners
        corners_1, ids_1, rejectedImgPoints_1 = aruco.detectMarkers(gray_1, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
        corners_2, ids_2, rejectedImgPoints_2 = aruco.detectMarkers(gray_2, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

        if len(corners_1) != 0 and len(corners_2) != 0:
            # Refine detected markers
            # Eliminates markers not part of our board, adds missing markers to the board
            corners_1, ids_1, rejectedImgPoints_1, recoveredIds_1 = aruco.refineDetectedMarkers(
                image=gray_1,
                board=CHARUCO_BOARD,
                detectedCorners=corners_1,
                detectedIds=ids_1,
                rejectedCorners=rejectedImgPoints_1,
                cameraMatrix=cam1_mtx,
                distCoeffs=distCoeffs)
            corners_2, ids_2, rejectedImgPoints_2, recoveredIds_2 = aruco.refineDetectedMarkers(
                image=gray_2,
                board=CHARUCO_BOARD,
                detectedCorners=corners_2,
                detectedIds=ids_2,
                rejectedCorners=rejectedImgPoints_2,
                cameraMatrix=cam2_mtx,
                distCoeffs=distCoeffs)

            # Only try to find CharucoBoard if we found markers
            if ids_1 is not None and ids_2 is not None and len(ids_1) > 10 and len(ids_2) > 10:
                # Get charuco corners and ids from detected aruco markers
                response_1, charuco_corners_1, charuco_ids_1 = aruco.interpolateCornersCharuco(
                    markerCorners=corners_1,
                    markerIds=ids_1,
                    image=gray_1,
                    board=CHARUCO_BOARD)

                response_2, charuco_corners_2, charuco_ids_2 = aruco.interpolateCornersCharuco(
                    markerCorners=corners_2,
                    markerIds=ids_2,
                    image=gray_2,
                    board=CHARUCO_BOARD)

                if response_1 is not None and response_2 is not None and response_1 > 20 and response_2 > 20\
                        and len(charuco_corners_1) == len(objp) and len(charuco_corners_2) == len(objp):
                    common_img_count += 1

                    objpoints.append(objp)
                    imgpoints_1.append(charuco_corners_1)
                    imgpoints_2.append(charuco_corners_2)

                    # Outline all of the markers detected in our image
                    img_1 = aruco.drawDetectedMarkers(img_1, corners_1, borderColor=(0, 0, 255))
                    img_2 = aruco.drawDetectedMarkers(img_2, corners_2, borderColor=(0, 0, 255))

                    # # Draw and display the corners
                    # images_display = np.hstack((img_1, img_2))
                    # cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
                    # cv2.imshow('RealSense', images_display)
                    # cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("common_img_count: ", common_img_count)

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    # transform coordinates in 1st cam frame to 2nd cam frame
    # gives the position of the 1st cam w.r.t the 2nd cam frame

    try:
        rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_1, imgpoints_2, cam1_mtx, distCoeffs,
                                                          cam2_mtx, distCoeffs, [640, 480], criteria_stereo, flags)
    # skip useless transformation
    except:
        continue

    print("Stereo Calibration RMS for cam " + str(x) + " and cam " + str(y) + ": ", rms)
    T = T.tolist()
    T = [T[0][0],T[1][0],T[2][0]]

    trans = np.vstack( (np.hstack((R, np.array(T).reshape(-1,1))), [0,0,0,1]) )
    inv_trans = np.linalg.inv(trans)  # transform coordinates in 2nd cam frame to 1st cam frame
    # gives the position of the 2nd cam w.r.t the 1st cam frame

    r_mat = inv_trans[0:3, 0:3]
    r = Rotation.from_matrix(r_mat)
    angle = r.as_euler('xyz', degrees=True)
    print('angle: ', angle)
    t = inv_trans[0:3, 3]
    print('translation: ', t)


    configuration_parameters["cams"][serial_numbers[x]][serial_numbers[y]] = {}
    configuration_parameters["cams"][serial_numbers[x]][serial_numbers[y]]["translation"] = trans[0:3, 3].tolist()
    configuration_parameters["cams"][serial_numbers[x]][serial_numbers[y]]["rotation"] = trans[0:3, 0:3].tolist()

    configuration_parameters["cams"][serial_numbers[y]][serial_numbers[x]] = {}
    configuration_parameters["cams"][serial_numbers[y]][serial_numbers[x]]["translation"] = t.tolist()
    configuration_parameters["cams"][serial_numbers[y]][serial_numbers[x]]["rotation"] = r_mat.tolist()

    json.dump(configuration_parameters, open("configuration_parameters.json", "w"), indent=4)

    # write_to_file('odometry.log', 0, np.eye(3), [0, 0, 0])
    # write_to_file('odometry.log', 1, r_mat, t)
