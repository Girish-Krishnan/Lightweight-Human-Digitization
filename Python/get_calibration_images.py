##################################################

# Take 12 images of a chessboard from each camera's perspective
# and save them in the calibration_images/ directory

##################################################


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

pipeline_D435.start(config_D435)

pipeline_D415 = rs.pipeline()
config_D415 = rs.config()
config_D415.enable_device('828612060381')
config_D415.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config_D415.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipeline_D415.start(config_D415)


"""
TAKE IMAGES USING THE CAMS

"""

image_count = 0

try:
    while True:

        # Camera 1
        
        frames_1 = pipeline_D415.wait_for_frames()
        color_frame_1 = frames_1.get_color_frame()
        if not color_frame_1:
            continue

        # Convert images to numpy arrays
        color_image_1 = np.asanyarray(color_frame_1.get_data())

        # Camera 2
        
        frames_2 = pipeline_D435.wait_for_frames()
        color_frame_2 = frames_2.get_color_frame()
        if not color_frame_2:
            continue
       
        # Convert images to numpy arrays

        color_image_2 = np.asanyarray(color_frame_2.get_data())
        # Stack all images horizontally
        images = np.hstack((color_image_1, color_image_2))

        # Show images from both cameras
        cv.namedWindow('RealSense', cv.WINDOW_NORMAL)
        cv.imshow('RealSense', images)

        # Save images and depth maps from both cameras by pressing spacebar
        ch = cv.waitKey(1)
        if ch==32:
            image_count +=1
            cv.imwrite("calibration_images/D415/" + str(image_count) + ".jpg",color_image_1)
            cv.imwrite("calibration_images/D435/" + str(image_count) + ".jpg",color_image_2)
            print("Saved image " + str(image_count))
            
            if image_count == 15:
                break


finally:

    # Stop streaming
    pipeline_D415.stop()
    pipeline_D435.stop()
    cv.destroyAllWindows()

