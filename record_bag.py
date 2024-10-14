import pyrealsense2 as rs
import time
from datetime import datetime

# Duration of the recording in seconds
RECORD_DURATION = 5

# Initialize the context to find devices
context = rs.context()

# List to store pipeline objects for each camera
pipelines = []

# Get the list of connected devices
devices = context.query_devices()

if len(devices) == 0:
    print("No Intel RealSense devices connected.")
    exit()

# Iterate through all connected devices
for i, device in enumerate(devices):
    serial_number = device.get_info(rs.camera_info.serial_number)
    print(f"Found camera with serial number: {serial_number}")
    
    # Create a pipeline for each device
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable the device by serial number
    config.enable_device(serial_number)
    
    # Enable streams for both color and depth
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Define the output file name for the bag file, including timestamp and serial number
    bag_filename = f"camera_{serial_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bag"
    config.enable_record_to_file(bag_filename)
    
    # Start the pipeline
    pipeline.start(config)
    
    # Store the pipeline for later use
    pipelines.append(pipeline)

print(f"Recording started for {len(devices)} camera(s).")

# Record for the specified duration (5 seconds)
time.sleep(RECORD_DURATION)

# Stop all pipelines simultaneously
for pipeline in pipelines:
    pipeline.stop()

print(f"Recording completed. {RECORD_DURATION} seconds of data recorded from each camera.")
