import ctypes
import threading
import numpy as np

# Load the C extension
capture = ctypes.CDLL('./capture.so')

depth_height = 480
depth_width = 640

# Define a function to capture images from one camera
def capture_from_camera(device_id):
    capture.capture_color_depth_images(device_id)

# Start 4 threads to capture images from each camera simultaneously
threads = []
for i in range(4):
    t = threading.Thread(target=capture_from_camera, args=(i,))
    threads.append(t)
    t.start()

# Wait for all threads to finish
for t in threads:
    t.join()

# Load the captured images into Numpy arrays
depth_images = []
for i in range(4):
    depth_image = np.fromfile("depth{}.bin".format(i), dtype=np.float32)
    depth_image = depth_image.reshape((depth_height, depth_width))
    depth_images.append(depth_image)
