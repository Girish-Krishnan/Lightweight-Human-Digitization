# Import required modules
from Reconstruction import Camera
import pyrealsense2 as rs
import os
import shutil
import argparse

if __name__ == '__main__':


    SERIAL_NUMBERS = []    # Initialize list of serial numbers
    
    parser = argparse.ArgumentParser(description='Capture images from Intel RealSense cameras')
    parser.add_argument('--hardware_reset', action='store_true', help='Reset all connected cameras')
    parser.add_argument('--data_reset', action='store_true', help='Delete all captured data')
    parser.add_argument('--output_dir', type=str, default='./Capture_Data', help='Output directory for captured data')
    parser.add_argument('-w', '--width', type=int, default=640, help='Width of captured images')
    parser.add_argument('-ht', '--height', type=int, default=480, help='Height of captured images')
    parser.add_argument('-f', '--fps', type=int, default=60, help='FPS of captured images')
    parser.add_argument('--warmup-frames', type=int, default=1000, help='Number of frames to capture for warm-up')
    parser.add_argument('-n', '--num-captures', type=int, default=20, help='Number of images to capture')
    args = parser.parse_args()

    # Check the values of the arguments passed
    if args.width < 0:
        print("Error: Width cannot be negative")
        exit(-1)
    if args.height < 0:
        print("Error: Height cannot be negative")
        exit(-1)
    if args.fps < 0:
        print("Error: FPS cannot be negative")
        exit(-1)
    if args.warmup_frames < 0:
        print("Error: Warmup frames cannot be negative")
        exit(-1)
    if args.num_captures < 0:
        print("Error: Number of captures cannot be negative")
        exit(-1)


    OUTPUT_DIR = args.output_dir
    RECONST_IMAGES_DIR = '/reconstruction_images'
    CALIB_IMAGES_DIR = '/calibration_images'

    numbered = False
    if args.num_captures > 1:
        numbered = True
 
    # Handle hardware reset option - resets all connected cameras
    # Use this option if the cameras are not responding
    if args.hardware_reset:
        print("Resetting all camera hardware")
        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            dev.hardware_reset()

        print('------------------------------------')
        print("Resetting complete")
        print('Exiting program')
        print('------------------------------------')
        print('Please run the code again without the --hardware_reset flag')
        exit(0)

    # Handle data reset option
    if args.data_reset:
        print("Deleting all capturing data")
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Get RealSense context
    ctx = rs.context()

    # Check if any devices are connected
    if len(ctx.devices) > 0:

        # Loop through connected devices
        for device_num in range(len(ctx.devices)):
            
            # Get serial number of device
            serial_number = ctx.devices[device_num].get_info(rs.camera_info.serial_number)

            # Print status message and append serial number to list
            print('Found device: ', ctx.devices[device_num].get_info(rs.camera_info.name), ' ', serial_number)
            SERIAL_NUMBERS.append(serial_number)

            # Create required directories            
            if not os.path.exists(OUTPUT_DIR + '/' + serial_number):
                os.makedirs(OUTPUT_DIR + '/' + serial_number)
            
            if not os.path.exists(OUTPUT_DIR + '/' + serial_number + RECONST_IMAGES_DIR):
                os.makedirs(OUTPUT_DIR + '/' + serial_number + RECONST_IMAGES_DIR)

            if not os.path.exists(OUTPUT_DIR + '/' + serial_number + CALIB_IMAGES_DIR):
                os.makedirs(OUTPUT_DIR + '/' + serial_number + CALIB_IMAGES_DIR)

    else:
        # Print error message and exit program
        print("No Intel Device connected")
        exit(-1)

    # Perform synchronous capture
    cam_array = Camera.SynchronousCapture(SERIAL_NUMBERS, 
                                          width=args.width, 
                                          height=args.height, 
                                          fps=args.fps, 
                                          warmup_frames=args.warmup_frames, 
                                          output_dir=OUTPUT_DIR,
                                          sub_dir=RECONST_IMAGES_DIR,
                                          numbered=numbered
                                          )
    
    cam_array.capture(args.num_captures, save_captures=True)

    # Stop capturing
    cam_array.stop()
