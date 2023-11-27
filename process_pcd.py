"""
IMPORTS
"""

from trajectory_utils.trajectory_io import *
import numpy as np
import json
import open3d as o3d
import glob
import matplotlib.pyplot as plt

# Set open3d random seed
np.random.seed(42)

pcd = o3d.io.read_point_cloud("./point_cloud_combined_3.ply")
# Remove outliers
# pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
# pcd, ind = pcd.remove_radius_outlier(nb_points=20, radius=0.05)

pcd.estimate_normals()
pcd.orient_normals_towards_camera_location()
pcd.normalize_normals()
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
densities = np.asarray(densities)
density_threshold = np.percentile(densities, 1)
vertices_to_remove = densities < density_threshold
mesh.remove_vertices_by_mask(vertices_to_remove)
mesh.compute_vertex_normals()
mesh.compute_triangle_normals()
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_duplicated_vertices()
mesh.remove_non_manifold_edges()
mesh.remove_unreferenced_vertices()
mesh = mesh.filter_smooth_simple(number_of_iterations=1)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))

kdtree = o3d.geometry.KDTreeFlann(pcd)


num_neighbors = 15
plane_threshold = 0.0001 


filtered_points = []

for i in range(len(pcd.points)):
    [k, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], num_neighbors)
    
    if k < num_neighbors:
        filtered_points.append(np.asarray(pcd.points[i]))
        continue

    neighbors = np.asarray(pcd.points)[idx, :]
    centroid = np.mean(neighbors, axis=0)
    neighbors_centered = neighbors - centroid
    _, s, _ = np.linalg.svd(neighbors_centered)

    if s[-1] < plane_threshold:
        continue
    else:
        filtered_points.append(np.asarray(pcd.points[i]))

filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
filtered_pcd, ind = filtered_pcd.remove_radius_outlier(nb_points=20, radius=0.007)

o3d.io.write_point_cloud("./pcd_combined_accessory_3.ply", filtered_pcd)

filtered_pcd_points = np.asarray(filtered_pcd.points)
mesh_points = np.asarray(mesh.vertices)

# COnsider only the mesh points whose coordinates are close to at least one point in filtered_pcd by a distance of 0.01
# This is to remove the points that are not in the filtered_pcd
new_mesh_points = []
for i in range(len(mesh_points)):
    if np.min(np.linalg.norm(filtered_pcd_points - mesh_points[i], axis=1)) < 0.1:
        new_mesh_points.append(mesh_points[i])

mesh_points = np.asarray(new_mesh_points)

mesh.vertices = o3d.utility.Vector3dVector(mesh_points)

o3d.io.write_triangle_mesh("./mesh_combined_accessory_3.ply", mesh)