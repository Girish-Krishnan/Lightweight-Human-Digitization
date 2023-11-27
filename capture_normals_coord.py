import pyrealsense2 as rs
import open3d as o3d
import numpy as np

# Start RealSense pipeline
pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth)
profile = pipe.start(config)

try:
    frames = pipe.wait_for_frames()
    depth_frame = frames.get_depth_frame()

    # Get intrinsic parameters
    intrinsics = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()

    # Generate point cloud
    pc = rs.pointcloud()
    pc.map_to(frames.get_color_frame())
    points = pc.calculate(depth_frame)

    # Get vertex and normal arrays
    vertices = np.asanyarray(points.get_vertices())
    normals = np.asanyarray(points.get_normals())

    # Map each 3D point to 2D
    depth_pixel_coords = []
    for vertex in vertices:
        x, y, z = vertex
        # Use intrinsic parameters to project 3D point to 2D
        pixel_x, pixel_y = rs.rs2_project_point_to_pixel(intrinsics, [x, y, z])
        depth_pixel_coords.append((pixel_x, pixel_y))

    # Now, `depth_pixel_coords` contains 2D coordinates on depth map corresponding to each 3D point
    # ...

finally:
    pipe.stop()
