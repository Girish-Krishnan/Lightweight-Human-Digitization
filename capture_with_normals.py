import pyrealsense2 as rs
import open3d as o3d
import numpy as np

pc = rs.pointcloud()

points = rs.points()

pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth)

pipe.start(config)

colorizer = rs.colorizer()

try:
    frames = pipe.wait_for_frames()
    colorized = colorizer.process(frames)

    ply = rs.save_to_ply("normal.ply")
    ply.set_option(rs.save_to_ply.option_ply_binary, False)
    ply.set_option(rs.save_to_ply.option_ply_normals, True)
    print("Saving...")
    ply.process(colorized)
    print("Done")

    pcd = o3d.io.read_triangle_mesh("normal.ply")
    o3d.visualization.draw_geometries([pcd])
    print(np.asanyarray(pcd.vertex_normals))

finally:
    pipe.stop()