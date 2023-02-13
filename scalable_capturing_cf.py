import pyrealsense2 as rs
import os
import cv2 as cv
import numpy as np
import json
import argparse
import shutil
import concurrent.futures

parser = argparse.ArgumentParser(description='Capture images for calibration')
parser.add_argument('--hardware_reset',help='reset all camera hardware')
parser.add_argument('--data_reset',help='delete all capturing data')
args = parser.parse_args()
serial_numbers = []
pipelines = []
configs = []
align = []
profiles = []

IMG_SIZE = [640, 480]

if args.hardware_reset:
    print("Resetting all camera hardware")
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        dev.hardware_reset()

    print("Resetting complete")
    print('Exiting program')
    print('------------------------------------')
    print('Please run the code again without the --hardware_reset flag')
    exit()


with(open("./configuration_parameters.json")) as f:
    configuration_parameters = json.load(f)
    num_images = configuration_parameters["num_calibration_imgs"]
    f.close()

ctx = rs.context()
if len(ctx.devices) > 0:

    for device_num in range(len(ctx.devices)):
        
        print ('Found device: ', ctx.devices[device_num].get_info(rs.camera_info.name), ' ', ctx.devices[device_num].get_info(rs.camera_info.serial_number))
        serial_numbers.append(ctx.devices[device_num].get_info(rs.camera_info.serial_number))
        pipelines.append((rs.pipeline(), device_num))
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

        # configs[device_num].enable_record_to_file('./' + ctx.devices[device_num].get_info(rs.camera_info.serial_number) + '/video.bag')

        # Align objects
        align_to = rs.stream.depth  # align to depth frame
        align.append(rs.align(align_to))
        pipelines[device_num][0].start(configs[device_num])

        # enable IR emitter and auto exposure
        profile = pipelines[device_num][0].get_active_profile()

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

    json.dump(configuration_parameters, open("configuration_parameters.json", "w"), indent = 4)    

else:

    print("No Intel Device connected")
    exit(-1)



"""
START RECORDING SOME FRAMES

"""


# Define a function to capture image and depth map for each pipeline

def capture_frame(pipeline,i):

    raw_color_images = len(serial_numbers) * [0]
    raw_color_frames = len(serial_numbers) * [0]
    color_images = len(serial_numbers) * [0]
    color_frames = len(serial_numbers) * [0]
    depth_images = len(serial_numbers) * [0]
    depth_frames = len(serial_numbers) * [0]
    depth_colormaps = len(serial_numbers) * [0]
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            raw_color_frames[i] = frames.get_color_frame()
            aligned_frames = align[i].process(frames)
            color_frames[i] = aligned_frames.get_color_frame()
            depth_frames[i] = aligned_frames.get_depth_frame()
            timestamp = depth_frames[i].get_timestamp()
            if not color_frames[i] or not depth_frames[i]:
                continue

            raw_color_images[i] = np.asanyarray(raw_color_frames[i].get_data())
            color_images[i] = np.asanyarray(color_frames[i].get_data())
            depth_images[i] = np.asanyarray(depth_frames[i].get_data())
            depth_colormaps[i] = cv.applyColorMap(cv.convertScaleAbs(depth_images[i], alpha=0.03), cv.COLORMAP_JET)
            
            # Display images using imshow

            # if (i == len(serial_numbers) - 1):
            #         stacked_color_images = np.hstack(tuple(color_images))
            #         stacked_depth_images = np.hstack(tuple(depth_colormaps))
            #         images = np.vstack((stacked_color_images, stacked_depth_images))
            #         cv.namedWindow('RealSense', cv.WINDOW_NORMAL)
            #         cv.imshow('RealSense', images)
            
            # ch = cv.waitKey(1)
            # if ch==32: 
                
            depth_frames[i] = rs.decimation_filter(1).process(depth_frames[i])
            depth_frames[i] = rs.disparity_transform(True).process(depth_frames[i])
            depth_frames[i] = rs.spatial_filter().process(depth_frames[i])
            depth_frames[i] = rs.temporal_filter().process(depth_frames[i])
            depth_frames[i] = rs.disparity_transform(False).process(depth_frames[i])

            raw_color_images[i] = np.asanyarray(raw_color_frames[i].get_data())
            color_images[i] = np.asanyarray(color_frames[i].get_data())
            depth_images[i] = np.asanyarray(depth_frames[i].get_data())

            depth_colormaps[i] = cv.applyColorMap(cv.convertScaleAbs(depth_images[i], alpha=0.03), cv.COLORMAP_JET)

            cv.imwrite('./' + serial_numbers[i] + '/sample_images/raw_image_cf.jpg', raw_color_images[i])
            cv.imwrite('./' + serial_numbers[i] + '/sample_images/image_cf.jpg', color_images[i])
            np.save('./' + serial_numbers[i] + '/sample_images/depth_map_cf.npy', depth_images[i])
            cv.imwrite('./' + serial_numbers[i] + '/sample_images/depth_cf.png', depth_colormaps[i])
            break
        
        
        # if i == len(serial_numbers) - 1:
        #     cv.destroyAllWindows()

        pipeline.stop()
    
    except Exception as e:
        print(e)



# Create a process pool to capture image and depth map from all pipelines
with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    results = [executor.submit(capture_frame, pipeline, device_id) for pipeline, device_id in pipelines]
    concurrent.futures.wait(results)