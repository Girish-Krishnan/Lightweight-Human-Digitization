import pyrealsense2 as rs
import numpy as np
import cv2

# Define the bag file path
bag_file_path = 'camera_828612060381_20240930_091608.bag'

# Create a pipeline object
pipeline = rs.pipeline()

# Create a config object
config = rs.config()

# Tell config that we will use a recorded .bag file as the input
config.enable_device_from_file(bag_file_path, repeat_playback=False)

# Start streaming from the bag file
profile = pipeline.start(config)

# Retrieve device information
device = profile.get_device()
serial_number = device.get_info(rs.camera_info.serial_number)

# Get the playback object
playback = profile.get_device().as_playback()

# Set playback to non-real-time to process the entire file
playback.set_real_time(False)

# Initialize variables for counting frames and calculating time
total_frames = 0
duration_in_seconds = 0

# Create an align object to align depth to color frame
align = rs.align(rs.stream.color)

# Iterate over the frames and display color and depth

while True:
        
        frames = pipeline.wait_for_frames()

        if not frames:
            continue
        
        # Align depth frame to color frame
        aligned_frames = align.process(frames)

        # Get color and depth frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Apply color map to depth image for visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Display images
        cv2.imshow('Color and Depth Frames', images)

        # Count frames
        total_frames += 1

        # Get timestamp of the current frame
        frame_time = frames.get_timestamp()
        duration_in_seconds = frame_time / 1000  # Convert milliseconds to seconds

        # Press 'q' to exit the loop manually
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Stop the pipeline and close OpenCV windows
pipeline.stop()
cv2.destroyAllWindows()

# Print out the information
print(f"Camera Serial Number: {serial_number}")
print(f"Total Time Duration (seconds): {duration_in_seconds:.2f}")
print(f"Total Number of Frames: {total_frames}")
