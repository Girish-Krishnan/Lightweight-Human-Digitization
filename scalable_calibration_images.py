import pyrealsense2 as rs
import os
import cv2 as cv
import numpy as np
import json
import cv2.aruco as aruco
import glob
from itertools import combinations

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

serial_numbers = []
pipelines = []
configs = []
profiles = []
num_images = 0

with(open("./configuration_parameters.json")) as f:
    configuration_parameters = json.load(f)
    num_images = configuration_parameters["num_calibration_imgs"]
    f.close()

NUM_CALIB_IMGS = configuration_parameters["num_calibration_imgs"]
CHECKERBOARD = (
configuration_parameters["checkerboard_dimensions"][0], configuration_parameters["checkerboard_dimensions"][1])
CHECKERBOARD_SIZE = configuration_parameters["checkerboard_size_mm"]  # units: millimeters
CHECKERBOARD_SIZE *= 0.001
IMAGE_TYPE = configuration_parameters["img_file_type"]
NUM_CAMS = len(configuration_parameters["cams"])
THRESHOLD = configuration_parameters["threshold"]

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
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
    #print('cam1: ', serial_numbers[x])
    #print('cam2: ', serial_numbers[y])

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
    imgpoints_1 = []

# RUN THIS ONCE AND THEN COMMENT IT
#
# ctx = rs.context()
# devices = ctx.query_devices() 
# for dev in devices:
#     dev.hardware_reset()
serial_numbers = []
ctx = rs.context()
if len(ctx.devices) > 0:

    for device_num in range(len(ctx.devices)):
        print ('Found device: ', ctx.devices[device_num].get_info(rs.camera_info.name), ' ', ctx.devices[device_num].get_info(rs.camera_info.serial_number))
        #serial_numbers.append(ctx.devices[device_num].get_info(rs.camera_info.serial_number))
        pipelines.append(rs.pipeline())
        configs.append(rs.config())
        configs[device_num].enable_device(ctx.devices[device_num].get_info(rs.camera_info.serial_number))
        # configs[device_num].enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        configs[device_num].enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        configs[device_num].enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)
        # configs[device_num].enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
        pipelines[device_num].start(configs[device_num])

        profile = pipelines[device_num].get_active_profile()
        profiles.append(profile)
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()
        # depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        # depth_intrinsics = depth_profile.get_intrinsics()
        ir_profile = rs.video_stream_profile(profile.get_stream(rs.stream.infrared))
        ir_intrinsics = ir_profile.get_intrinsics()

        s_num = ctx.devices[device_num].get_info(rs.camera_info.serial_number)
        serial_numbers.append(s_num)
        configuration_parameters["cams"][s_num]["intrinsics"] = {}
        configuration_parameters["cams"][s_num]["intrinsics"]["img_size"] = [640, 480]
        configuration_parameters["cams"][s_num]["intrinsics"]["focal_length"] = [color_intrinsics.fx,
                                                                                       color_intrinsics.fy]
        configuration_parameters["cams"][s_num]["intrinsics"]["img_center"] = [color_intrinsics.ppx,
                                                                                     color_intrinsics.ppy]
        # configuration_parameters["cams"][s_num]["intrinsics"]["depth_focal_length"] = [depth_intrinsics.fx,
        #                                                                                depth_intrinsics.fy]
        # configuration_parameters["cams"][s_num]["intrinsics"]["depth_img_center"] = [depth_intrinsics.ppx,
        #                                                                              depth_intrinsics.ppy]
        configuration_parameters["cams"][s_num]["intrinsics"]["ir_focal_length"] = [ir_intrinsics.fx,
                                                                                       ir_intrinsics.fy]
        configuration_parameters["cams"][s_num]["intrinsics"]["ir_img_center"] = [ir_intrinsics.ppx,
                                                                                     ir_intrinsics.ppy]
        assert color_intrinsics.coeffs == [0.0, 0.0, 0.0, 0.0, 0.0]
        # assert depth_intrinsics.coeffs == [0.0, 0.0, 0.0, 0.0, 0.0]
        assert ir_intrinsics.coeffs == [0.0, 0.0, 0.0, 0.0, 0.0]

        # disable IR emitter and auto exposure
        device = profile.get_device()
        depth_sensor = device.query_sensors()[0]
        emitter = depth_sensor.get_option(rs.option.emitter_enabled)
        print("old emitter = ", emitter)
        depth_sensor.set_option(rs.option.emitter_enabled, 0)  # disable IR emitter
        emitter1 = depth_sensor.get_option(rs.option.emitter_enabled)
        print("new emitter = ", emitter1)
        depth_sensor.set_option(rs.option.enable_auto_exposure, False)  # disable auto exposure

        if not os.path.exists(ctx.devices[device_num].get_info(rs.camera_info.serial_number)):
          # Create a new directory because it does not exist 
            os.makedirs(ctx.devices[device_num].get_info(rs.camera_info.serial_number))

        if not os.path.exists(ctx.devices[device_num].get_info(rs.camera_info.serial_number) + "/sample_images"):
            os.makedirs(ctx.devices[device_num].get_info(rs.camera_info.serial_number) + "/sample_images")
        if not os.path.exists(ctx.devices[device_num].get_info(rs.camera_info.serial_number) + "/calibration_images"):
            os.makedirs(ctx.devices[device_num].get_info(rs.camera_info.serial_number) + "/calibration_images")

    json.dump(configuration_parameters, open("configuration_parameters.json", "w"), indent=4)
else:

    print("No Intel Device connected")
    exit(-1)



"""
START RECORDING SOME FRAMES

"""

color_images = len(serial_numbers) * [0]
ir_images = len(serial_numbers) * [0]
ir_images_processed = len(serial_numbers) * [0] 
image_count = 0
exposure_d415 = 70000
gain_d415 = 30

set_gain = False

for i in range(len(serial_numbers)):

    sensor = profiles[i].get_device().query_sensors()[0]
    #print(serial_numbers[i])

    if set_gain == False:
        try:
            sensor.set_option(rs.option.gain, gain_d415)
        finally:
            set_gain = True
    
    sensor.set_option(rs.option.exposure, exposure_d415)

try:
    while True:

        for i in range(len(serial_numbers)):

            # sensor = profiles[i].get_device().query_sensors()[0]
            # #print(serial_numbers[i])
        
            # if set_gain == False:
            #     try:
            #         sensor.set_option(rs.option.gain, gain_d415)
            #     finally:
            #         set_gain = True
            
            # sensor.set_option(rs.option.exposure, exposure_d415)

            frames = pipelines[i].wait_for_frames()

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_images[i] = np.asanyarray(color_frame.get_data())

            ir_frame = frames.first(rs.stream.infrared)
            if not ir_frame:
                continue

            ir_frame_original = np.asanyarray(ir_frame.get_data())
            ir_frame_processed = np.copy(color_images[i])
            corners_1, ids_1, rejectedImgPoints_1 = aruco.detectMarkers(ir_frame_original, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
            #print(len(corners_1))
            if len(corners_1) != 0:
            # Refine detected markers
            # Eliminates markers not part of our board, adds missing markers to the board
                corners_1, ids_1, rejectedImgPoints_1, recoveredIds_1 = aruco.refineDetectedMarkers(
                image=ir_frame_original,
                board=CHARUCO_BOARD,
                detectedCorners=corners_1,
                detectedIds=ids_1,
                rejectedCorners=rejectedImgPoints_1,
                cameraMatrix=cam1_mtx,
                distCoeffs=distCoeffs)
                
            # Only try to find CharucoBoard if we found markers
                if ids_1 is not None and len(ids_1) > 10:
                # Get charuco corners and ids from detected aruco markers
                    response_1, charuco_corners_1, charuco_ids_1 = aruco.interpolateCornersCharuco(
                    markerCorners=corners_1,
                    markerIds=ids_1,
                    image=ir_frame_original,
                    board=CHARUCO_BOARD)
                    

                    if response_1 is not None and response_1 > 20\
                        and len(charuco_corners_1) == len(objp):

                        objpoints.append(objp)
                        imgpoints_1.append(charuco_corners_1)

                        # Outline all of the markers detected in our image
                        ir_frame_processed = aruco.drawDetectedMarkers(ir_frame_processed, corners_1, borderColor=(0, 0, 255))
                        


            ir_images[i] = ir_frame_original
            ir_images_processed[i] = ir_frame_processed

        # Stack all images horizontally
        # images_color = np.hstack(tuple(color_images))
        images_ir = np.hstack(tuple(ir_images_processed))

        # Show images from all cameras
        cv.namedWindow('RealSense', cv.WINDOW_NORMAL)
        cv.imshow('RealSense', images_ir)
        ch = cv.waitKey(1)
        if ch==32:
            image_count +=1
            print("Saving image: ", image_count)
            for i in range(len(serial_numbers)):
                cv.imwrite('./' + serial_numbers[i] + '/calibration_images/image_' + str(image_count) + '.jpg', ir_images[i])

            if image_count == num_images:
                break

finally:

    # Stop streaming
    for pipeline in pipelines:
        pipeline.stop()

    cv.destroyAllWindows()
