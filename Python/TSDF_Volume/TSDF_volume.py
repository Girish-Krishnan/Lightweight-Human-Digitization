import cv2
import matplotlib.pyplot as plt
from utils.trajectory_io import *
import open3d as o3d
import json

camera_poses = read_trajectory("odometry.log")
SETTINGS_PATH = '../configuration_parameters.json'
param = json.load(open(SETTINGS_PATH))
cams_list = list(param["cams"].keys())

# TSDF volume
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=2 / 512.0,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

# show RGBD images from all the views
for i, cam in enumerate(cams_list):
    # if i == 0: continue
    # print("Integrate {:d}-th image into the volume.".format(i))
    color = o3d.io.read_image(cam + "../sample_images/image.jpg")
    depth = o3d.io.read_image(cam + "../sample_images/depth.png")

    depth_image = np.asanyarray(depth)
    color_image = np.asanyarray(color)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.09), cv2.COLORMAP_JET)
    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape

    # If depth and color resolutions are different, resize color image to match depth image for display
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                         interpolation=cv2.INTER_AREA)
        images = np.hstack((resized_color_image, depth_colormap))
    else:
        images = np.hstack((color_image, depth_colormap))

    plt.imshow(images)
    plt.show()

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)

    intr = o3d.camera.PinholeCameraIntrinsic(
        width=640,
        height=480,
        fx=param['cams'][cam]['intrinsics']['ir_focal_length'][0],
        fy=param['cams'][cam]['intrinsics']['ir_focal_length'][1],
        cx=param['cams'][cam]['intrinsics']['ir_img_center'][0],
        cy=param['cams'][cam]['intrinsics']['ir_img_center'][1]
    )

    volume.integrate(rgbd, intr, np.linalg.inv(camera_poses[i].pose))
    # break

# point cloud generation
pcd = volume.extract_point_cloud()
downpcd = pcd.voxel_down_sample(voxel_size=0.0005)
o3d.visualization.draw_geometries([downpcd])
coordinates = np.asarray(downpcd.points)