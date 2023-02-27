# Implement scalable_capturing.py using multiprocessing for parallelism
# and threading for concurrency.

import argparse
from Camera import Camera
import pyrealsense2 as rs
import os
import shutil
import time

if __name__ == '__main__':

    SERIAL_NUMBERS = []

    parser = argparse.ArgumentParser(description='Capture images')
    parser.add_argument('--hardware_reset',help='reset all camera hardware')
    parser.add_argument('--data_reset',help='delete all capturing data')
    args = parser.parse_args()

    if args.hardware_reset:
        print("Resetting all camera hardware")
        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            dev.hardware_reset()

        print("Resetting complete")
        print('Exiting program')
        print('------------------------------------')
        print('Please run the code again without the --hardware_reset flag')
        exit()

    ctx = rs.context()
    if len(ctx.devices) > 0:

        for device_num in range(len(ctx.devices)):
            
            serial_number = ctx.devices[device_num].get_info(rs.camera_info.serial_number)
            print ('Found device: ', ctx.devices[device_num].get_info(rs.camera_info.name), ' ', serial_number)

            SERIAL_NUMBERS.append(serial_number)
            
            
            if args.data_reset:
                print("Deleting all capturing data")
                if os.path.exists(serial_number):
                    shutil.rmtree(serial_number)

            if not os.path.exists(serial_number):
                os.makedirs(serial_number)
            
            if not os.path.exists(serial_number + "/sample_images"):
                os.makedirs(serial_number + "/sample_images")

            if not os.path.exists(serial_number + "/calibration_images"):
                os.makedirs(serial_number + "/calibration_images")   

    else:

        print("No Intel Device connected")
        exit(-1)

    # Perform synchronous capture

    cam_array = Camera.SynchronousCapture(SERIAL_NUMBERS)
    # Measure time taken to capture
    cam_array.capture()
    cam_array.save()
    cam_array.stop()

    

