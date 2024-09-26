import open3d as o3d
import numpy as np

def find_overlapping_points(pcd1, pcd2, distance_threshold):
    """
    Find the indices of points in overlapping regions between two point clouds.
    Args:
        pcd1: First Open3D point cloud.
        pcd2: Second Open3D point cloud.
        distance_threshold: Distance below which points are considered to overlap.
    Returns:
        Indices of overlapping points in pcd1 and pcd2.
    """
    # KDTree for fast nearest neighbor search
    kdtree = o3d.geometry.KDTreeFlann(pcd2)

    overlap_indices_pcd1 = []
    overlap_indices_pcd2 = []

    for i, point in enumerate(pcd1.points):
        [_, idx, dist] = kdtree.search_knn_vector_3d(point, 1)  # Search for the nearest neighbor
        if dist[0] < distance_threshold:
            overlap_indices_pcd1.append(i)
            overlap_indices_pcd2.append(idx[0])

    return overlap_indices_pcd1, overlap_indices_pcd2

def average_colors_in_overlap(pcd1, pcd2, overlap_indices_pcd1, overlap_indices_pcd2):
    """
    Average the colors of overlapping points between two point clouds.
    Args:
        pcd1: First Open3D point cloud.
        pcd2: Second Open3D point cloud.
        overlap_indices_pcd1: Indices of overlapping points in pcd1.
        overlap_indices_pcd2: Indices of overlapping points in pcd2.
    Returns:
        Updated point clouds with averaged colors in overlapping regions.
    """
    colors_pcd1 = np.asarray(pcd1.colors)
    colors_pcd2 = np.asarray(pcd2.colors)

    # Average the colors in the overlapping regions
    for idx1, idx2 in zip(overlap_indices_pcd1, overlap_indices_pcd2):
        avg_color = (colors_pcd1[idx1] + colors_pcd2[idx2]) / 2
        colors_pcd1[idx1] = avg_color
        colors_pcd2[idx2] = avg_color

    # Update the colors of the point clouds
    pcd1.colors = o3d.utility.Vector3dVector(colors_pcd1)
    pcd2.colors = o3d.utility.Vector3dVector(colors_pcd2)

    return pcd1, pcd2

# Load or create your point clouds (pcd1, pcd2, pcd3, pcd4)
# Replace this with your actual point cloud loading code
pcd1 = o3d.io.read_point_cloud("DATASET/Subject_21/828612060381/individual_pcd_trans.ply")
pcd2 = o3d.io.read_point_cloud("DATASET/Subject_21/839112060979/individual_pcd_trans.ply")
pcd3 = o3d.io.read_point_cloud("DATASET/Subject_21/839112061696/individual_pcd_trans.ply")
pcd4 = o3d.io.read_point_cloud("DATASET/Subject_21/839212060064/individual_pcd_trans.ply")

# Define the distance threshold for detecting overlapping points
distance_threshold = 0.02  # Adjust based on your data

# Find overlapping points between each pair of point clouds
overlap_idx_12_pcd1, overlap_idx_12_pcd2 = find_overlapping_points(pcd1, pcd2, distance_threshold)
overlap_idx_13_pcd1, overlap_idx_13_pcd3 = find_overlapping_points(pcd1, pcd3, distance_threshold)
overlap_idx_14_pcd1, overlap_idx_14_pcd4 = find_overlapping_points(pcd1, pcd4, distance_threshold)
overlap_idx_23_pcd2, overlap_idx_23_pcd3 = find_overlapping_points(pcd2, pcd3, distance_threshold)
overlap_idx_24_pcd2, overlap_idx_24_pcd4 = find_overlapping_points(pcd2, pcd4, distance_threshold)
overlap_idx_34_pcd3, overlap_idx_34_pcd4 = find_overlapping_points(pcd3, pcd4, distance_threshold)

# Average colors in overlapping areas
pcd1, pcd2 = average_colors_in_overlap(pcd1, pcd2, overlap_idx_12_pcd1, overlap_idx_12_pcd2)
pcd1, pcd3 = average_colors_in_overlap(pcd1, pcd3, overlap_idx_13_pcd1, overlap_idx_13_pcd3)
pcd1, pcd4 = average_colors_in_overlap(pcd1, pcd4, overlap_idx_14_pcd1, overlap_idx_14_pcd4)
pcd2, pcd3 = average_colors_in_overlap(pcd2, pcd3, overlap_idx_23_pcd2, overlap_idx_23_pcd3)
pcd2, pcd4 = average_colors_in_overlap(pcd2, pcd4, overlap_idx_24_pcd2, overlap_idx_24_pcd4)
pcd3, pcd4 = average_colors_in_overlap(pcd3, pcd4, overlap_idx_34_pcd3, overlap_idx_34_pcd4)

# Combine the point clouds back into one
combined_pcd = pcd1 + pcd2 + pcd3 + pcd4

# Visualize the result
o3d.visualization.draw_geometries([combined_pcd])

original_combined_pcd = o3d.io.read_point_cloud("point_cloud_combined.ply")
o3d.visualization.draw_geometries([original_combined_pcd])
