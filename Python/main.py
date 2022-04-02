"""
IMPORTS
"""
from DigitizeLib import Camera

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

image_name = "0500"
cam_2.add_image(image_name)
# cam_1.add_image(image_name)
# cam_2.add_image(image_name)
# cam_3.add_image(image_name)

cam_2.get_RGBD()
cam_2.point_cloud()
cam_2.visualize()