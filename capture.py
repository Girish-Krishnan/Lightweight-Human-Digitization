# Import required modules
import argparse
from Reconstruction import Camera
import pyrealsense2 as rs
import os
import shutil


if __name__ == '__main__':

    # Initialize list of serial numbers
    SERIAL_NUMBERS = []

    # Create parser object for command line arguments
    parser = argparse.ArgumentParser(description='Capture images')
    parser.add_argument('--hardware_reset', help='reset all camera hardware')
    parser.add_argument('--data_reset', help='delete all capturing data')
    args = parser.parse_args()

    # Handle hardware reset option
    if args.hardware_reset:
        print("Resetting all camera hardware")
        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            dev.hardware_reset()

        # Print status messages and exit program
        print("Resetting complete")
        print('Exiting program')
        print('------------------------------------')
        print('Please run the code again without the --hardware_reset flag')
        exit()

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
            
            # Handle data reset option
            if args.data_reset:
                print("Deleting all capturing data")
                if os.path.exists("./Camera_Data/" + serial_number):
                    shutil.rmtree("./Camera_Data/" + serial_number)

            # Create required directories
            if not os.path.exists("./Camera_Data/" + serial_number):
                os.makedirs("./Camera_Data/" + serial_number)
            
            if not os.path.exists("./Camera_Data/" + serial_number + "/sample_images"):
                os.makedirs("./Camera_Data/" + serial_number + "/sample_images")

            if not os.path.exists("./Camera_Data/" + serial_number + "/calibration_images"):
                os.makedirs("./Camera_Data/" + serial_number + "/calibration_images")   

    else:
        # Print error message and exit program
        print("No Intel Device connected")
        exit(-1)

    # Perform synchronous capture
    cam_array = Camera.SynchronousCapture(SERIAL_NUMBERS)

    # Start capturing buffers of 3 frames
    cam_array.capture_buffer(20)

    # single frame
    cam_array.capture(20)

    # Save captured images
    cam_array.save()
    # Stop capturing
    cam_array.stop()
