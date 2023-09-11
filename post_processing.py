"""
IMPORTS
"""

from trajectory_utils.trajectory_io import *
import numpy as np
import json
import open3d as o3d
import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import cv2


"""
GET CONFIGS
"""

def dynamic_thickness(y, max_thickness, height, min_y, max_y):
    # This function returns a thickness based on y-coordinate
    # You can modify this function to suit your specific requirements
    return max_thickness if y > min_y + (max_y - min_y) / 25 else max_thickness // 3

def draw_contour_with_dynamic_thickness(img, contour, max_thickness):
    height, _, = img.shape
    min_y = np.min(contour[:, :, 1])
    max_y = np.max(contour[:, :, 1])
    for i in range(len(contour) - 1):
        y_value = contour[i][0][1]
        thickness = dynamic_thickness(y_value, max_thickness, height, min_y, max_y)
        
        # Drawing a line segment between two subsequent points
        cv2.line(img, tuple(contour[i][0]), tuple(contour[i+1][0]), (0, 255, 0), thickness)
    
    # Closing the contour (connecting the last and the first point)
    y_value = contour[-1][0][1]
    thickness = dynamic_thickness(y_value, max_thickness, height, min_y, max_y)
    cv2.line(img, tuple(contour[-1][0]), tuple(contour[0][0]), (0, 255, 0), thickness)

def remove_border(image, border_thickness):
    # Threshold the grayscale image to create a binary mask
    _, binary_mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Find the contours of the human subject
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contour with a thickness to create a border
    #cv2.drawContours(binary_mask, contours, -1, (0, 0, 0), border_thickness)
    draw_contour_with_dynamic_thickness(binary_mask, contours[0], border_thickness)

    return binary_mask

SETTINGS_PATH = './configuration_parameters.json'
param = json.load(open(SETTINGS_PATH))
cams_list = list(param["cams"].keys())
print("cams_list: ", cams_list)
camera_poses = read_trajectory("odometry.log")

num_images = len(glob.glob('./Capture_Data/' + cams_list[0] + "/reconstruction_images/image.jpg"))
print("num_images: ", num_images)

for j in range(num_images):
    # TSDF volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=2 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    
    for i, cam in enumerate(cams_list):

        color = o3d.io.read_image('./Capture_Data/' + cam + f"/reconstruction_images/image.jpg")
        depth_map_data = np.load('./Capture_Data/' + cam + f"/reconstruction_images/depth_map.npy")
        # Segment the depth map, and removing pixels that are too far away or too close
        depth_map_data[depth_map_data > 1500] = 0
        depth_map_data[depth_map_data < 500] = 0

        # Use PIL Filter to Filter the image
        depth_map_data_rgb = Image.fromarray(depth_map_data)
        # Convert to RGB
        depth_map_data_rgb = depth_map_data_rgb.convert("RGB")
        depth_map_data_rgb = depth_map_data_rgb.filter(ImageFilter.ModeFilter(size=13))

        # filter depth map to include points where depth_map_rgb is white
        depth_map_data_rgb = np.array(depth_map_data_rgb)

        # Convert to grayscale
        depth_map_data_rgb = depth_map_data_rgb[:, :, 0]

        # Remove the border of the object
        depth_map_data_rgb = remove_border(depth_map_data_rgb, 20)

        # Remove regions in the resulting image which have a large variance in depth
        window_size = 5
        variance_threshold = 5
        variance = cv2.filter2D(depth_map_data_rgb.astype(float)**2, -1, np.ones((window_size, window_size)), borderType=cv2.BORDER_REFLECT)
        variance_mask = variance < variance_threshold
        variance_copy = np.copy(depth_map_data_rgb)
        variance_copy[variance_mask] = 255
        
        # Bitwise-AND variance_copy and depth_map_data_rgb
        depth_map_data_rgb = cv2.bitwise_and(variance_copy, depth_map_data_rgb)

        depth_map_data[depth_map_data_rgb == 0] = 0

        depth = o3d.geometry.Image(depth_map_data)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)

        intr = o3d.camera.PinholeCameraIntrinsic(
            width=param['cams'][cam]['intrinsics']['img_size'][0],
            height=param['cams'][cam]['intrinsics']['img_size'][1],
            fx=param['cams'][cam]['intrinsics']['ir_focal_length'][0],
            fy=param['cams'][cam]['intrinsics']['ir_focal_length'][1],
            cx=param['cams'][cam]['intrinsics']['ir_img_center'][0],
            cy=param['cams'][cam]['intrinsics']['ir_img_center'][1]
        )

        volume.integrate(rgbd, intr, np.linalg.inv(camera_poses[i].pose))

    # point cloud generation
    pcd = volume.extract_point_cloud()
    pcd = pcd.voxel_down_sample(voxel_size=0.0005)
    pcd.estimate_normals()

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 2 * avg_dist

    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
         pcd, o3d.utility.DoubleVector([radius, radius * 2]))
    
    # # Decimate mesh
    # dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)
    # dec_mesh.remove_degenerate_triangles()
    # dec_mesh.remove_duplicated_triangles()
    # dec_mesh.remove_duplicated_vertices()
    # dec_mesh.remove_non_manifold_edges()
    # dec_mesh.remove_unreferenced_vertices()

    # # Smooth mesh
    # dec_mesh = dec_mesh.filter_smooth_simple(number_of_iterations=1)
    # dec_mesh = o3d.t.geometry.TriangleMesh.from_legacy(dec_mesh).fill_holes().to_legacy()

    # Remove areas of the mesh that are disconnected from the main body
    

    o3d.visualization.draw_geometries([pcd])
    break

    o3d.io.write_point_cloud(f"volumes/volume_{j}.ply", pcd)
    print(f"volume_{j}.ply saved.")

