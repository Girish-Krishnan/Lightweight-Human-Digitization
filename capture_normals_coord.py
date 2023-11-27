import pyrealsense2 as rs
import open3d as o3d
import numpy as np

# Start RealSense pipeline
pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth)
config.enable_stream(rs.stream.color)
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

    ply = rs.save_to_ply("normal.ply")
    ply.set_option(rs.save_to_ply.option_ply_binary, False)
    ply.set_option(rs.save_to_ply.option_ply_normals, True)
    ply.process(frames)

    # Get vertex and normal arrays
    vertices = np.asanyarray(points.get_vertices())
    mesh = o3d.io.read_triangle_mesh("normal.ply")
    normals = np.asanyarray(mesh.vertex_normals)
    print(vertices.shape)
    print(normals.shape)

    # Map each 3D point to 2D
    depth_pixel_coords = []
    for vertex in vertices:
        x, y, z = vertex
        # Use intrinsic parameters to project 3D point to 2D
        pixel_x, pixel_y = rs.rs2_project_point_to_pixel(intrinsics, [x, y, z])
        if np.isnan(pixel_x):
            continue
        depth_pixel_coords.append((int(pixel_x), int(pixel_y)))

    print(len(depth_pixel_coords))

finally:
    pipe.stop()
