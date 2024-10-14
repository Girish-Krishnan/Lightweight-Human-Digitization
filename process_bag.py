
import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Directory to store processed data
output_directory = "output_data"

# List of bag files to process
bag_files = [
    "camera_828612060381_20240930_091608.bag",
    "camera_839112060979_20240930_091607.bag",
    "camera_839112061696_20240930_091608.bag",
    "camera_839212060064_20240930_091607.bag"
]

# Frame interval in seconds
FRAME_INTERVAL = 0.5

def create_directory_structure(base_dir, serial_number, recording_time):
    # Create the main directory for the recording time
    recording_dir = os.path.join(base_dir, f"Recording_time_{recording_time}")
    os.makedirs(recording_dir, exist_ok=True)

    # Create subdirectories for the camera serial number
    camera_dir = os.path.join(recording_dir, serial_number, "reconstruction_images")
    os.makedirs(camera_dir, exist_ok=True)

    return camera_dir

def process_bag_file(bag_file_path, serial_number, base_output_dir):
    # Create pipeline and config
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file_path, repeat_playback=False)

    # Start pipeline
    profile = pipeline.start(config)

    # Align depth to color
    align = rs.align(rs.stream.color)

    # Set the playback to non-real-time to process frames as fast as possible
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    recording_time = 0
    last_frame_time = 0

    try:
        while True:
            # Wait for the next set of frames
            frames = pipeline.wait_for_frames()

            # Get the timestamp of the current frame
            frame_time = frames.get_timestamp() / 1000.0  # Convert to seconds

            # Check if it's time to save a frame based on the interval
            if frame_time - last_frame_time >= FRAME_INTERVAL:
                # Align the frames
                aligned_frames = align.process(frames)

                # Get color and depth frames
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # Convert color image to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                # Create directory structure for this recording time
                camera_output_dir = create_directory_structure(base_output_dir, serial_number, recording_time)

                # Save color image as image.jpg
                image_filename = os.path.join(camera_output_dir, "image.jpg")
                cv2.imwrite(image_filename, color_image)

                # Save depth image as depth.npy
                depth_filename = os.path.join(camera_output_dir, "depth.npy")
                np.save(depth_filename, depth_image)

                # Print the timestamps of the saved frame
                color_timestamp = color_frame.get_timestamp() / 1000.0  # Convert to seconds
                depth_timestamp = depth_frame.get_timestamp() / 1000.0  # Convert to seconds

                print(f"Saved frame {recording_time} from camera {serial_number}")
                print(f"Color frame timestamp: {color_timestamp:.3f} seconds")
                print(f"Depth frame timestamp: {depth_timestamp:.3f} seconds")

                # Update the last frame time and recording time counter
                last_frame_time = frame_time
                recording_time += 1

    except RuntimeError:
        # This will happen when the playback reaches the end of the file
        print(f"Reached end of bag file for camera {serial_number}")
    
    finally:
        # Stop the pipeline when done
        pipeline.stop()

def main():
    # Create the base output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Iterate over each bag file and process it
    for bag_file in bag_files:
        # Extract serial number from the file name (assuming the format)
        serial_number = bag_file.split('_')[1]
        print(f"Processing bag file: {bag_file} for camera: {serial_number}")
        
        # Process each bag file
        process_bag_file(bag_file, serial_number, output_directory)

if __name__ == "__main__":
    main()
