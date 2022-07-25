import pyrealsense2 as rs
import os
import cv2 as cv
import numpy as np
import json

serial_numbers = []
pipelines = []
configs = []

ctx = rs.context()
if len(ctx.devices) > 0:

    for device_num in range(len(ctx.devices)):
        print ('Found device: ', ctx.devices[device_num].get_info(rs.camera_info.name), ' ', ctx.devices[device_num].get_info(rs.camera_info.serial_number))
        serial_numbers.append(ctx.devices[device_num].get_info(rs.camera_info.serial_number))
        pipelines.append(rs.pipeline())
        configs.append(rs.config())
        configs[device_num].enable_device(ctx.devices[device_num].get_info(rs.camera_info.serial_number))
        configs[device_num].enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        configs[device_num].enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # Align objects
        align_to = rs.stream.color
        align = rs.align(align_to)
        pipelines[device_num].start(configs[device_num])

        if not os.path.exists(ctx.devices[device_num].get_info(rs.camera_info.serial_number)):
            os.makedirs(ctx.devices[device_num].get_info(rs.camera_info.serial_number))
        
        if not os.path.exists(ctx.devices[device_num].get_info(rs.camera_info.serial_number) + "/sample_images"):
            os.makedirs(ctx.devices[device_num].get_info(rs.camera_info.serial_number) + "/sample_images")
        if not os.path.exists(ctx.devices[device_num].get_info(rs.camera_info.serial_number) + "/calibration_images"):
            os.makedirs(ctx.devices[device_num].get_info(rs.camera_info.serial_number) + "/calibration_images")

else:

    print("No Intel Device connected")
    exit(-1)



"""
START RECORDING SOME FRAMES

"""

color_images = len(serial_numbers) * [0]
depth_images = len(serial_numbers) * [0]

try:
    while True:

        for i in range(len(serial_numbers)):
        
            frames = pipelines[i].wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Convert images to numpy arrays
            color_images[i] = np.asanyarray(color_frame.get_data())
            depth_images[i] = np.asanyarray(depth_frame.get_data())

        # Stack all images horizontally
        images = np.hstack(tuple(color_images))

        # Show images from both cameras
        cv.namedWindow('RealSense', cv.WINDOW_NORMAL)
        cv.imshow('RealSense', images)
        ch = cv.waitKey(1)
        if ch==32:

            for i in range(len(serial_numbers)):
                cv.imwrite('./' + serial_numbers[i] + '/sample_images/image.jpg', color_images[i])
                np.save('./' + serial_numbers[i] + '/sample_images/depth_map.npy', depth_images[i])
        
            break


finally:

    # Stop streaming
    for pipeline in pipelines:
        pipeline.stop()

    cv.destroyAllWindows()


# ######################################################################################################################

with(open("./configuration_parameters.json")) as f:
    configuration_parameters = json.load(f)
    for i in serial_numbers:
        configuration_parameters["cams"][i] = {}

    json.dump(configuration_parameters, open("configuration_parameters.json", "w"), indent = 4)
    f.close()