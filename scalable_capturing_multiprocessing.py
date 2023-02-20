# Implement scalable_capturing.py using multiprocessing for parallelism
# and threading for concurrency.
# This is a solution to the problem described in the README.md file.
#

import pyrealsense2 as rs
import os
import cv2 as cv
import numpy as np
import json
import shutil
import argparse
import multiprocessing

parser = argparse.ArgumentParser(description='Capture images for calibration')
parser.add_argument('--hardware_reset',help='reset all camera hardware')
parser.add_argument('--data_reset',help='delete all capturing data')
args = parser.parse_args()

def record_frames(device_num):
        
            serial_numbers = []
            pipelines = []
            configs = []
            align = []
            profiles = []
            ctx = rs.context()

            IMG_SIZE = [640, 480]

            print ('Found device: ', ctx.devices[device_num].get_info(rs.camera_info.name), ' ', ctx.devices[device_num].get_info(rs.camera_info.serial_number))
            serial_numbers.append(ctx.devices[device_num].get_info(rs.camera_info.serial_number))
            pipelines.append(rs.pipeline())
            configs.append(rs.config())
            configs[device_num].enable_device(ctx.devices[device_num].get_info(rs.camera_info.serial_number))
            configs[device_num].enable_stream(rs.stream.depth, 640,480, rs.format.z16, 60)
            configs[device_num].enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 60)
            configs[device_num].enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 60)
            #os.chmod(ctx.devices[device_num].get_info(rs.camera_info.serial_number), 0o777)
            
            if args.data_reset:
                print("Deleting all capturing data")
                if os.path.exists(ctx.devices[device_num].get_info(rs.camera_info.serial_number)):
                    shutil.rmtree(ctx.devices[device_num].get_info(rs.camera_info.serial_number))

            if not os.path.exists(ctx.devices[device_num].get_info(rs.camera_info.serial_number)):
                os.makedirs(ctx.devices[device_num].get_info(rs.camera_info.serial_number))
            
            if not os.path.exists(ctx.devices[device_num].get_info(rs.camera_info.serial_number) + "/sample_images"):
                os.makedirs(ctx.devices[device_num].get_info(rs.camera_info.serial_number) + "/sample_images")
            if not os.path.exists(ctx.devices[device_num].get_info(rs.camera_info.serial_number) + "/calibration_images"):
                os.makedirs(ctx.devices[device_num].get_info(rs.camera_info.serial_number) + "/calibration_images")

            configs[device_num].enable_record_to_file('./' + ctx.devices[device_num].get_info(rs.camera_info.serial_number) + '/video.bag')

            # Align objects
            align_to = rs.stream.depth  # align to depth frame
            align.append(rs.align(align_to))
            pipelines[device_num].start(configs[device_num])

            # enable IR emitter and auto exposure
            profile = pipelines[device_num].get_active_profile()

            profiles.append(profile)
            color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
            color_intrinsics = color_profile.get_intrinsics()
            # depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
            # depth_intrinsics = depth_profile.get_intrinsics()
            ir_profile = rs.video_stream_profile(profile.get_stream(rs.stream.infrared))
            ir_intrinsics = ir_profile.get_intrinsics()

            s_num = ctx.devices[device_num].get_info(rs.camera_info.serial_number)
            # configuration_parameters["cams"][s_num] = {}
            # configuration_parameters["cams"][s_num]["intrinsics"] = {}
            # configuration_parameters["cams"][s_num]["intrinsics"]["img_size"] = IMG_SIZE
            # configuration_parameters["cams"][s_num]["intrinsics"]["focal_length"] = [color_intrinsics.fx,
            #                                                                                color_intrinsics.fy]
            # configuration_parameters["cams"][s_num]["intrinsics"]["img_center"] = [color_intrinsics.ppx,
            #                                                                              color_intrinsics.ppy]
            # # configuration_parameters["cams"][s_num]["intrinsics"]["depth_focal_length"] = [depth_intrinsics.fx,
            # #                                                                                depth_intrinsics.fy]
            # # configuration_parameters["cams"][s_num]["intrinsics"]["depth_img_center"] = [depth_intrinsics.ppx,
            # #                                                                              depth_intrinsics.ppy]
            # configuration_parameters["cams"][s_num]["intrinsics"]["ir_focal_length"] = [ir_intrinsics.fx,
            #                                                                                ir_intrinsics.fy]
            # configuration_parameters["cams"][s_num]["intrinsics"]["ir_img_center"] = [ir_intrinsics.ppx,
            #                                                                              ir_intrinsics.ppy]

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

            # Wait for a coherent pair of frames: depth and color
            frames = pipelines[device_num].wait_for_frames()
            aligned_frames = align[device_num].process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                return
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Save color as .jpg and depth as .npy
            cv.imwrite(ctx.devices[device_num].get_info(rs.camera_info.serial_number) + "/sample_images/image.jpg", color_image)
            np.save(ctx.devices[device_num].get_info(rs.camera_info.serial_number) + "/sample_images/depth_map.npy", depth_image)

if __name__ == '__main__':
    ctx = rs.context()
    if args.hardware_reset:
        print("Resetting all camera hardware")
        
        devices = ctx.query_devices()
        for dev in devices:
            dev.hardware_reset()

        print("Resetting complete")
        print('Exiting program')
        print('------------------------------------')
        print('Please run the code again without the --hardware_reset flag')
        exit()

    
    if len(ctx.devices) == 0:
        print("No Intel Device connected")
        exit(-1)    
        
    """
    START RECORDING SOME FRAMES
    """

    with open('configuration_parameters.json') as f:
        configuration_parameters = json.load(f)
        serial_numbers = list(configuration_parameters["cams"].keys())
        f.close()

    # Now we will start streaming with the default streams of color and depth in parallel for all cameras
    # We will also create an align object to align the depth frames to color frames
    # Use multiprocessing

    # Create a pool of processes. By default, one is created for each CPU in your machine.
    pool = multiprocessing.Pool(processes=len(serial_numbers))


    # Start the processes and store them in a list
    processes = [pool.apply_async(record_frames, args=(i,)) for i in range(len(serial_numbers))]    

    # Get the results from the processes. This will block until all results are in
    results = [p.get() for p in processes]

    # Close the pool and wait for the work to finish
    pool.close()

