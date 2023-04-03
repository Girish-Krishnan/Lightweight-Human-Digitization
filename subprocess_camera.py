from Reconstruction import Camera
import argparse
import pyrealsense2 as rs

parser = argparse.ArgumentParser(description='Capture images')
# Add one required argument for the serial number
parser.add_argument('--serial_number', help='serial number of camera', required=True)
args = parser.parse_args()

# Create camera object
camera = Camera.RealSenseCamera(args.serial_number)

# Capture images
frames = camera.get_frames()

# Print timestamp of frames
print("Timestamp of frames:")
print(frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival))

camera.process_frames(frames)
camera.save_frames()