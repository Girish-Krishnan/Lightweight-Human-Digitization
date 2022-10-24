import pyrealsense2 as rs
import os
import cv2 as cv
import numpy as np
import json

serial_numbers = []
pipelines = []
configs = []
align = []

# RUN THIS ONCE AND THEN COMMENT IT OUT
#
# ctx = rs.context()
# devices = ctx.query_devices()
# for dev in devices:
#     dev.hardware_reset()

ctx = rs.context()
if len(ctx.devices) > 0:

    for device_num in range(len(ctx.devices)):
        print ('Found device: ', ctx.devices[device_num].get_info(rs.camera_info.name), ' ', ctx.devices[device_num].get_info(rs.camera_info.serial_number))
        serial_numbers.append(ctx.devices[device_num].get_info(rs.camera_info.serial_number))
        pipelines.append(rs.pipeline())
        configs.append(rs.config())
        configs[device_num].enable_device(ctx.devices[device_num].get_info(rs.camera_info.serial_number))
        configs[device_num].enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)
        configs[device_num].enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
        configs[device_num].enable_record_to_file('./' + ctx.devices[device_num].get_info(rs.camera_info.serial_number) + '/video.bag')

        # Align objects
        align_to = rs.stream.depth  # align to depth frame
        align.append(rs.align(align_to))
        pipelines[device_num].start(configs[device_num])

        # enable IR emitter and auto exposure
        profile = pipelines[device_num].get_active_profile()
        device = profile.get_device()
        depth_sensor = device.query_sensors()[0]
        emitter = depth_sensor.get_option(rs.option.emitter_enabled)
        print("old emitter = ", emitter)
        depth_sensor.set_option(rs.option.emitter_enabled, 1)  # enable IR emitter
        emitter1 = depth_sensor.get_option(rs.option.emitter_enabled)
        print("new emitter = ", emitter1)
        depth_sensor.set_option(rs.option.enable_auto_exposure, True)  # enable auto exposure

        depth_sensor.set_option(rs.option.laser_power, 360)  # max laser power
        print("laser power: ", depth_sensor.get_option(rs.option.laser_power))

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
raw_color_images = len(serial_numbers) * [0]
color_images = len(serial_numbers) * [0]
depth_images = len(serial_numbers) * [0]
depth_colormaps = len(serial_numbers) * [0]

try:
    while True:

        for i in range(len(serial_numbers)):

            frames = pipelines[i].wait_for_frames()
            raw_color_frame = frames.get_color_frame()
            aligned_frames = align[i].process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Convert images to numpy arrays
            raw_color_images[i] = np.asanyarray(raw_color_frame.get_data())
            color_images[i] = np.asanyarray(color_frame.get_data())
            depth_images[i] = np.asanyarray(depth_frame.get_data())

            depth_colormaps[i] = cv.applyColorMap(cv.convertScaleAbs(depth_images[i], alpha=0.03), cv.COLORMAP_JET)

        # Stack all images horizontally
        stacked_color_images = np.hstack(tuple(color_images))
        stacked_depth_images = np.hstack(tuple(depth_colormaps))
        images = np.vstack((stacked_color_images, stacked_depth_images))
        cv.namedWindow('RealSense', cv.WINDOW_NORMAL)
        cv.imshow('RealSense', images)

        ch = cv.waitKey(1)
        if ch==32:

            for i in range(len(serial_numbers)):
                cv.imwrite('./' + serial_numbers[i] + '/sample_images/raw_image.jpg', raw_color_images[i])
                cv.imwrite('./' + serial_numbers[i] + '/sample_images/image.jpg', color_images[i])
                np.save('./' + serial_numbers[i] + '/sample_images/depth_map.npy', depth_images[i])
                cv.imwrite('./' + serial_numbers[i] + '/sample_images/depth.png', depth_colormaps[i])
            break


finally:

    # Stop streaming
    for pipeline in pipelines:
        pipeline.stop()

    cv.destroyAllWindows()


######################################################################################################################

with(open("./configuration_parameters.json")) as f:
    configuration_parameters = json.load(f)
    for i in serial_numbers:
        configuration_parameters["cams"][i] = {}

    json.dump(configuration_parameters, open("configuration_parameters.json", "w"), indent = 4)
    f.close()