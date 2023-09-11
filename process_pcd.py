"""
IMPORTS
"""

from trajectory_utils.trajectory_io import *
import numpy as np
import json
import open3d as o3d
import glob
import matplotlib.pyplot as plt

"""
GET CONFIGS
"""

pcd = o3d.io.read_point_cloud("./point_cloud_combined.ply")
points = np.asarray(pcd.points)
# Filter out points, based on x, y, z coordinates
pcd = pcd.select_by_index(
    np.where(
        np.logical_and(
            np.logical_and(points[:, 0] > -0.5, points[:, 0] < 0.5),
            np.logical_and(points[:, 1] > -0.5, points[:, 1] < 0.5),
            np.logical_and(points[:, 2] > 0, points[:, 2] > 0)
        )
    )[0]
)

# Apply Poisson surface reconstruction
pcd.estimate_normals()

# Save pcd
o3d.io.write_point_cloud("./point_cloud_filtered.ply", pcd)
    
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