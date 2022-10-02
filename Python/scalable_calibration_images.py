import pyrealsense2 as rs
import os
import cv2 as cv
import numpy as np
import json

serial_numbers = []
pipelines = []
configs = []
profiles = []
num_images = 0

with(open("./configuration_parameters.json")) as f:
    configuration_parameters = json.load(f)
    num_images = configuration_parameters["num_calibration_imgs"]
    f.close()

# RUN THIS ONCE AND THEN COMMENT IT
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
image_count = 0

exposure_d415 = 90000
gain_d415 = 50
exposure_d435 = 8000
gain_d435 = 20

try:
    while True:

        for i in range(len(serial_numbers)):

            sensor = profiles[i].get_device().query_sensors()[0]
            #print(serial_numbers[i])
            if serial_numbers[i] == "819312073170":
                sensor.set_option(rs.option.gain, gain_d435)
                sensor.set_option(rs.option.exposure, exposure_d435)
            else:
                # sensor.set_option(rs.option.gain, gain_d415)
                sensor.set_option(rs.option.exposure, exposure_d415)

            frames = pipelines[i].wait_for_frames()

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_images[i] = np.asanyarray(color_frame.get_data())

            ir_frame = frames.first(rs.stream.infrared)
            if not ir_frame:
                continue
            ir_images[i] = np.asanyarray(ir_frame.get_data())

        # Stack all images horizontally
        # images_color = np.hstack(tuple(color_images))
        images_ir = np.hstack(tuple(ir_images))

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

