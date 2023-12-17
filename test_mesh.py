import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

pcd = o3d.io.read_point_cloud("27-11.ply")

# Use DBSCAN to remove clusters of points that are not part of the object, but retain the colors of the points that are kept
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
normals_original = np.asarray(pcd.normals) 

# DBSCAN clustering
clustering = DBSCAN(eps=0.02, min_samples=10).fit(points)
labels = clustering.labels_

# Identify the largest cluster
unique_labels, counts = np.unique(labels, return_counts=True)
largest_cluster_label = unique_labels[np.argmax(counts)]

# Filter out noise (small clusters)
filtered_points = points[labels == largest_cluster_label]
filtered_colors = colors[labels == largest_cluster_label]
filtered_normals = normals_original[labels == largest_cluster_label]

# Create a new point cloud with filtered data
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
filtered_pcd.normals = o3d.utility.Vector3dVector(filtered_normals)

pcd = filtered_pcd

# Remove points whose z value is too small
# pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 2] < -0.7)[0])

camera_location = np.array([0.0, 0.0, -0.93])
#camera_location = np.array([0.0, 0.0, 0.0])

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1000, max_nn=30))
# pcd.orient_normals_towards_camera_location(camera_location=camera_location)
pcd.orient_normals_consistent_tangent_plane(k=20)

# Reverse the direction of the normals, if the point has z value greater than the camera location
#pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals) * -1.0)

points = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)

# Now that we have the original normals and the computed normals, we can take proportionally from each
# to get the final normals
alpha = 0.2
normals = alpha * normals + (1 - alpha) * normals_original
pcd.normals = o3d.utility.Vector3dVector(normals)

# kdtree = o3d.geometry.KDTreeFlann(pcd)
# for i in range(len(points)):
#     _, idx, _ = kdtree.search_knn_vector_3d(pcd.points[i], 10)
#     normals[i] = np.mean(np.asarray(pcd.normals)[idx], axis=0)

# Create lines for normals from points to (points + normals)
lines = [[i, i + len(points)] for i in range(len(points))]
points_with_normals = np.vstack([points, points + 0.02*normals])  # Adjust the 0.02 scalar to scale the length of the normals

# Create a LineSet from the points and lines
line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_with_normals),
            lines=o3d.utility.Vector2iVector(lines),
    )

mesh, mesh_id = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0.0, scale=1.0, linear_fit=False)[0:2]
mesh.compute_vertex_normals()
mesh.orient_triangles()

# Run DBSCAN on the mesh to remove noise, similar to what we did with the point cloud
points = np.asarray(mesh.vertices)
normals_original = np.asarray(mesh.vertex_normals)
colors = np.asarray(mesh.vertex_colors)

# DBSCAN clustering
clustering = DBSCAN(eps=0.02, min_samples=30).fit(points)
labels = clustering.labels_

# Identify the largest cluster
unique_labels, counts = np.unique(labels, return_counts=True)
largest_cluster_label = unique_labels[np.argmax(counts)]

# Print the number of clusters and the points in each cluster
print("Number of clusters: " + str(len(unique_labels)))
for i in range(len(unique_labels)):
    print("Cluster " + str(i) + ": " + str(counts[i]))

# Filter out noise (small clusters)
filtered_points = points[labels == largest_cluster_label]
filtered_colors = colors[labels == largest_cluster_label]
filtered_normals = normals_original[labels == largest_cluster_label]

# Print the number of points in the filtered point cloud
print("Number of points in filtered point cloud: " + str(len(filtered_points)))

# Create a new point cloud with filtered data
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
filtered_pcd.normals = o3d.utility.Vector3dVector(filtered_normals)

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(filtered_pcd, depth=8, width=0.0, scale=1.0, linear_fit=False)[0]

# Visualize the point cloud and normals
o3d.visualization.draw_geometries([pcd, line_set, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=camera_location)])
o3d.visualization.draw_geometries([pcd, line_set])
o3d.visualization.draw_geometries([mesh])
o3d.io.write_triangle_mesh("realsense_normals.ply", mesh)