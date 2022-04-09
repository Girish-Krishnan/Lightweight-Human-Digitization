"""
IMPORTS
"""
from DigitizeLib import Camera
import numpy as np

"""
CREATING CAMERA OBJECTS
"""

cam_0 = Camera.Camera(0)
cam_1 = Camera.Camera(1)
cam_2 = Camera.Camera(2)
cam_3 = Camera.Camera(3)


"""
READING IMAGE FILE AND DEPTH MAP AS INPUTS...
"""

image_name = "0350"
cam_0.add_image(image_name)
cam_1.add_image(image_name)
cam_2.add_image(image_name)
cam_3.add_image(image_name)

cam_0.point_cloud()
cam_1.point_cloud()
cam_2.point_cloud()
cam_3.point_cloud()

cam_0.visualize()
cam_1.visualize()
cam_2.visualize()
cam_3.visualize()

combiner = Camera.Combiner()
combiner.add_image(image_name)
combiner.add()