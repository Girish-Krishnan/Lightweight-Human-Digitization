import numpy as np
import open3d as o3d

if __name__ == "__main__":
    pcds = []
    trajectory = o3d.io.read_pinhole_camera_trajectory("./data/THuman/camera_trajectory.txt")
    
    im1 = o3d.io.read_image("./sample_images/D415_sample.jpg")
    im2 = o3d.io.read_image("./sample_images/D415_sample.npy")
    im = o3d.geometry.RGBDImage.create_from_color_and_depth(
            im2, im1, 1000.0, 5.0, False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            im, trajectory.parameters[0].intrinsic,
            trajectory.parameters[0].extrinsic)
    pcds.append(pcd)

    im1 = o3d.io.read_image("./sample_images/D435_sample.jpg")
    im2 = o3d.io.read_image("./sample_images/D435_sample.npy")
    im = o3d.geometry.RGBDImage.create_from_color_and_depth(
            im2, im1, 1000.0, 5.0, False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            im, trajectory.parameters[1].intrinsic,
            trajectory.parameters[1].extrinsic)
    pcds.append(pcd)

    o3d.visualization.draw_geometries(pcds)