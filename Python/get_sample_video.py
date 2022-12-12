"""
To get a sample video, recorded from both cameras

"""

"""
IMPORTS
"""
from Camera import Camera
import numpy as np
import cv2 as cv
import pyrealsense2 as rs

"""
INITIALIZE 2 REALSENSE CAMERAS
"""

pipeline_D435 = rs.pipeline()
config_D435 = rs.config()
config_D435.enable_device('819312073170')
config_D435.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config_D435.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config_D435.enable_record_to_file('./sample_video/D435_sample.bag')
pipeline_D435.start(config_D435)

pipeline_D415 = rs.pipeline()
config_D415 = rs.config()
config_D415.enable_device('828612060381')
config_D415.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config_D415.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config_D415.enable_record_to_file('./sample_video/D415_sample.bag')
pipeline_D415.start(config_D415)


"""
START RECORDING SOME FRAMES

"""


try:
    while True:

        # Camera 1
        
        frames_1 = pipeline_D415.wait_for_frames()
        color_frame_1 = frames_1.get_color_frame()
        depth_frame_1 = frames_1.get_depth_frame()
        if not color_frame_1 or not depth_frame_1:
            continue

        # Convert images to numpy arrays
        color_image_1 = np.asanyarray(color_frame_1.get_data())
        depth_image_1 = np.asanyarray(depth_frame_1.get_data())

        # Camera 2
        
        frames_2 = pipeline_D435.wait_for_frames()
        color_frame_2 = frames_2.get_color_frame()
        depth_frame_2 = frames_2.get_depth_frame()
        if not color_frame_2 or not depth_frame_2:
            continue
       
        # Convert images to numpy arrays

        color_image_2 = np.asanyarray(color_frame_2.get_data())
        depth_image_2 = np.asanyarray(depth_frame_2.get_data())

        # Stack all images horizontally
        images = np.hstack((color_image_1, color_image_2))

        # Show images from both cameras
        cv.namedWindow('RealSense', cv.WINDOW_NORMAL)
        cv.imshow('RealSense', images)
        cv.waitKey(1)

        # Stop recording by pressing spacebar
        ch = cv.waitKey(1)
        if ch==32:
            cv.imwrite('./sample_images/D415_sample.jpg', color_image_1)
            np.save('./sample_images/D415_sample.npy', depth_image_1)
            cv.imwrite('./sample_images/D435_sample.jpg', color_image_2)
            np.save('./sample_images/D435_sample.npy', depth_image_2)
            break


finally:

    # Stop streaming
    pipeline_D415.stop()
    pipeline_D435.stop()
    cv.destroyAllWindows()

