#####################################################
##               Read bag from file                ##
#####################################################


# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

import json

import open3d as o3d

"""
GET CAM DATA
"""

SETTINGS_PATH = './configuration_parameters.json'
param = json.load(open(SETTINGS_PATH))
cams_list = list(param["cams"].keys())
print("cams_list: ", cams_list)
CAM_DATA = [param["cams"][cam] for cam in param["cams"]]  # camera data


video_paths = ['./'+cam+'/video.bag' for cam in cams_list]

for video_path in video_paths:

    try:
        # Create pipeline
        pipeline = rs.pipeline()

        # Create a config object
        config = rs.config()

        # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
        rs.config.enable_device_from_file(config, video_path)

        # Configure the pipeline to stream the depth stream
        # Change this parameters according to the recorded bag file resolution
        config.enable_stream(rs.stream.depth, rs.format.z16, 30)

        # Start streaming from file
        pipeline.start(config)

        # Create opencv window to render image in
        cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
        
        # Create colorizer object
        colorizer = rs.colorizer()

        # Streaming loop
        while True:
            # Get frameset of depth
            frames = pipeline.wait_for_frames()

            # Get depth frame
            depth_frame = frames.get_depth_frame()

            # Colorize depth frame to jet colormap
            depth_color_frame = colorizer.colorize(depth_frame)

            # Convert depth_frame to numpy array to render image in opencv
            depth_color_image = np.asanyarray(depth_color_frame.get_data())

            # Render image in opencv window
            cv2.imshow("Depth Stream", depth_color_image)
            key = cv2.waitKey(1)
            # if pressed escape exit program
            if key == 27:
                cv2.destroyAllWindows()
                break

    finally:
        pass


# visualization

# vis = o3d.visualization.Visualizer()
# vis.create_window()

# # geometry is the point cloud used in your animaiton
# geometry = o3d.geometry.PointCloud()
# vis.add_geometry(geometry)

# for i in range(icp_iteration):
#     # now modify the points of your geometry
#     # you can use whatever method suits you best, this is just an example
#     geometry.points = pcd_list[i].points
#     vis.update_geometry(geometry)
#     vis.poll_events()
#     vis.update_renderer()