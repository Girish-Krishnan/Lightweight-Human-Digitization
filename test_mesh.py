import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

pcd = o3d.io.read_point_cloud("pcd_combined_accessory_3.ply")
# Remove points whose z value is too small
# pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 2] < -0.7)[0])


camera_location = np.array([0.0, 0.0, -0.93])
#camera_location = np.array([0.0, 0.0, 0.0])

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1000, max_nn=30))
#pcd.orient_normals_towards_camera_location(camera_location=camera_location)
pcd.orient_normals_consistent_tangent_plane(k=20)

# Reverse the direction of the normals, if the point has z value greater than the camera location
#pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals) * -1.0)

points = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)

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
# orient mesh normals towards interior of mesh


# Visualize the point cloud and normals
o3d.visualization.draw_geometries([pcd, line_set, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=camera_location)])
o3d.visualization.draw_geometries([pcd, line_set])
o3d.visualization.draw_geometries([mesh])
o3d.io.write_triangle_mesh("mesh_combined_accessory_3.ply", mesh)