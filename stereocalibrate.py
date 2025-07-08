"""
FINDS THE CORRESPONDING CAMERA PARAMETERS FOR ANY ARBITRARY NUMBER OF CONNECTED CAMERAS

"""

import cv2
import numpy as np
import glob
import json
from itertools import combinations
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import traceback
import argparse
import matplotlib.pyplot as plt
from trajectory_io import *


RECONST_IMAGES_DIR = '/reconstruction_images'
CALIB_IMAGES_DIR = '/calibration_images'

parser = argparse.ArgumentParser(description='Stereo calibration')
parser.add_argument('--bundle_adjust',action='store_true')
parser.add_argument('--config_file', type=str, default='./configuration_parameters.json')
parser.add_argument('--charuco_rows', type=int, default=8)
parser.add_argument('--charuco_cols', type=int, default=11)
parser.add_argument('--square_length', type=float, default=0.060)
parser.add_argument('--marker_length', type=float, default=0.044)
parser.add_argument('--data_dir', type=str, default='./Capture_Data')
parser.add_argument('--display', action='store_true')
parser.add_argument('--width', type=int, default=640)
parser.add_argument('--height', type=int, default=480)
parser.add_argument('--threshold_charuco',type=int,default=20)

args = parser.parse_args()

# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_250)
CHARUCOBOARD_ROWCOUNT = args.charuco_rows + 1 # number of squares in the charuco board
CHARUCOBOARD_COLCOUNT = args.charuco_cols + 1

distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Create grid board object we're using in our stream
CHARUCO_BOARD = aruco.CharucoBoard_create(
    squaresX=CHARUCOBOARD_COLCOUNT,
    squaresY=CHARUCOBOARD_ROWCOUNT,
    squareLength=args.square_length,
    markerLength=args.marker_length,
    dictionary=ARUCO_DICT)


# LOAD INITIAL CONFIG PARAMS, which are then modified in the program
with(open(args.config_file)) as f:
    configuration_parameters = json.load(f)
    f.close()

NUM_CALIB_IMGS = configuration_parameters["num_calibration_imgs"]
CHECKERBOARD = (
configuration_parameters["checkerboard_dimensions"][0], configuration_parameters["checkerboard_dimensions"][1])
CHECKERBOARD_SIZE = configuration_parameters["checkerboard_size_mm"]  # units: millimeters
CHECKERBOARD_SIZE *= 0.001 # convert to meters
IMAGE_TYPE = configuration_parameters["img_file_type"]
NUM_CAMS = len(configuration_parameters["cams"])
THRESHOLD = configuration_parameters["threshold"]

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
serial_numbers = list(configuration_parameters["cams"].keys())

objp = np.zeros((CHECKERBOARD[1] * CHECKERBOARD[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[1], 0:CHECKERBOARD[0]].T.reshape(-1, 2)
objp = CHECKERBOARD_SIZE * objp

def compute_distance(op, left_mtx, left_dist, left_pts,
                     right_mtx, right_dist, right_pts, R, T):
    """compute the distance between each calibration target corner."""
    
    left, left_r, left_t = cv2.solvePnP(op, left_pts, left_mtx, left_dist, 0)
    right, right_r, right_t = cv2.solvePnP(op, right_pts, right_mtx, right_dist, 0)
    left_r = cv2.Rodrigues(left_r)[0]
    right_r = cv2.Rodrigues(right_r)[0]
    left_p = left_r.dot(op.T) + left_t
    right_p = right_r.dot(op.T) + right_t
    right_p = R.T.dot(right_p - T)
    return (np.abs(left_p - right_p) * 1000).ravel()  

def fun(parameters, obj_pts, left_pts, right_pts, left_mtx, left_dist, right_mtx, right_dist):
    """
    Compute residuals:
    `parameters` contains camera parameters and R and T.
    """
    r = cv2.Rodrigues(np.array(parameters[0:3]))[0]
    t = np.array(parameters[3:]).reshape(-1,1)
    residuals = []
    for i, (op, lp, rp) in enumerate(zip(obj_pts, left_pts, right_pts)):
        dist = compute_distance(op,left_mtx, left_dist, lp,
                                     right_mtx, right_dist, rp, r, t)
        residuals.append(dist)
        
    return np.hstack(residuals)

if NUM_CAMS == 0:
    print("No camera images found. Please check the directory.")
    exit(-1)

# Extracting path of individual image stored in a given directory
images = []
for i in range(NUM_CAMS):
    images.append(glob.glob(args.data_dir + "/" + serial_numbers[i] + CALIB_IMAGES_DIR + "/*" + IMAGE_TYPE))

image_pairs = combinations(range(NUM_CAMS), 2)  # finding all distinct pairs of cameras

for pair in image_pairs:
    x = pair[0]
    y = pair[1]
    common_img_count = 0  # the number of common images between the two cameras that contain the chessboard successfully detected

    print("Calibrating cameras " + str(serial_numbers[x]) + " and " + str(serial_numbers[y]))

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

                if response_1 is not None and response_2 is not None:
                    common_ids, idx_1, idx_2 = np.intersect1d(charuco_ids_1, charuco_ids_2, return_indices=True)
                    if len(common_ids) >= args.threshold_charuco:
                        common_img_count += 1

                        charuco_corners_1_matched = charuco_corners_1[idx_1]
                        charuco_corners_2_matched = charuco_corners_2[idx_2]
                        objp_matched = objp[common_ids.flatten()]

                        objpoints.append(objp_matched)
                        imgpoints_1.append(charuco_corners_1_matched)
                        imgpoints_2.append(charuco_corners_2_matched)

                        if args.display:
                            # Outline all of the markers detected in our image
                            img_1 = aruco.drawDetectedMarkers(img_1, corners_1, borderColor=(0, 0, 255))
                            img_2 = aruco.drawDetectedMarkers(img_2, corners_2, borderColor=(0, 0, 255))

                            # # Draw and display the corners
                            images_display = np.hstack((img_1, img_2))
                            cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
                            cv2.imshow('RealSense', images_display)
                            cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("Number of common images: ", common_img_count)

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    if common_img_count >= THRESHOLD:
        try:
            rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_1, imgpoints_2, cam1_mtx, distCoeffs,
                                                            cam2_mtx, distCoeffs, [args.width, args.height], criteria_stereo, flags)

            
            if args.bundle_adjust:
                r = cv2.Rodrigues(R)[0].flatten().tolist()
                t = T.flatten().tolist()

                params_rt = np.array(r + t)

                plt.plot(fun(params_rt, objpoints, imgpoints_1, imgpoints_2, cam1_mtx, distCoeffs, cam2_mtx, distCoeffs))

                res = least_squares(fun, params_rt, verbose=2, method ='trf', xtol=3e-16, ftol=3e-16, gtol=3e-16,
                            loss='soft_l1', f_scale=0.001,
                            args=(objpoints, imgpoints_1, imgpoints_2, cam1_mtx, distCoeffs, cam2_mtx, distCoeffs))
                
                plt.plot(res.fun)

                R2 = np.array(Rotation.from_euler('xyz',(res.x)[0:3],degrees=False).as_matrix())
                T2 = np.array((res.x)[3:]).reshape(-1,1)
            else:
                R2 = R
                T2 = T


        # skip useless transformation
        except:
            traceback.print_exc()
            continue
    else:
        continue

    print("Stereo Calibration RMS for cam " + str(x) + " and cam " + str(y) + ": ", rms)
    print()
    T = T.tolist()
    T = [T[0][0],T[1][0],T[2][0]]

    T2 = T2.tolist()
    T2 = [T2[0][0],T2[1][0],T2[2][0]]

    trans = np.vstack( (np.hstack((R, np.array(T).reshape(-1,1))), [0,0,0,1]) )
    inv_trans = np.linalg.inv(trans)  # transform coordinates in 2nd cam frame to 1st cam frame
    # gives the position of the 2nd cam w.r.t the 1st cam frame

    trans_2 = np.vstack( (np.hstack((R2, np.array(T2).reshape(-1,1))), [0,0,0,1]) )
    inv_trans_2 = np.linalg.inv(trans_2)  # transform coordinates in 2nd cam frame to 1st cam frame
    # gives the position of the 2nd cam w.r.t the 1st cam frame    

    r_mat = inv_trans[0:3, 0:3]
    t = inv_trans[0:3, 3]
    print('translation: ', t)


    configuration_parameters["cams"][serial_numbers[x]][serial_numbers[y]] = {}
    configuration_parameters["cams"][serial_numbers[x]][serial_numbers[y]]["translation"] = trans_2[0:3, 3].tolist()
    configuration_parameters["cams"][serial_numbers[x]][serial_numbers[y]]["rotation"] = trans_2[0:3, 0:3].tolist()

    configuration_parameters["cams"][serial_numbers[y]][serial_numbers[x]] = {}
    configuration_parameters["cams"][serial_numbers[y]][serial_numbers[x]]["translation"] = t.tolist()
    configuration_parameters["cams"][serial_numbers[y]][serial_numbers[x]]["rotation"] = r_mat.tolist()

    json.dump(configuration_parameters, open(args.config_file, "w"), indent=4)