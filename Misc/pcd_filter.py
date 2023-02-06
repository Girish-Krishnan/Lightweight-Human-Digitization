import numpy as np
import open3d as o3d

# Load the point cloud
pcd = o3d.io.read_point_cloud("point_cloud_combined.ply")

# Create a kd-tree for the point cloud
pcd_tree = o3d.geometry.KDTreeFlann(pcd)

# Initialize an empty point cloud for the filtered point cloud
filtered_pcd = o3d.geometry.PointCloud()

# Read the frames from the point cloud
for i in range(len(pcd.points)):
    # Apply averaging filter on the current point
    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 1000)
    filtered_point = np.average(np.asarray(pcd.points)[idx], axis=0)
    filtered_pcd.points.append(filtered_point)

filtered_pcd.colors = pcd.colors
# Save the filtered point cloud
o3d.io.write_point_cloud("filtered_point_cloud.ply", filtered_pcd)
o3d.visualization.draw_geometries([filtered_pcd])