import cv2, sys, pyrealsense2 as rs, numpy as np
import multiprocessing

# List to store pipeline objects and device IDs
pipelines = []
device_ids = []

# Initialize the pipeline and start streaming
for i, device in enumerate(rs.context().devices):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device.get_info(rs.camera_info.serial_number))
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    pipelines.append((pipeline, i))
    device_ids.append(device.get_info(rs.camera_info.serial_number))

# Define a function to capture image and depth map for each pipeline
def capture_frame(device_id):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device_id)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        cv2.imshow("Color Stream " + str(device_id), color_image)
        cv2.imshow("Depth Stream " + str(device_id), depth_image)

        key = cv2.waitKey(1)
        if key == 32: # Press spacebar to capture
            cv2.imwrite("color_" + str(device_id) + ".jpg", color_image)
            cv2.imwrite("depth_" + str(device_id) + ".jpg", depth_image)

# Create a process for each pipeline
processes = []
for device_id in device_ids:
    p = multiprocessing.Process(target=capture_frame, args=(device_id,))
    p.start()
    processes.append(p)

# Wait for all processes to finish
for p in processes:
    p.join()
