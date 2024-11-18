import pyrealsense2 as rs

# Function to pretty-print intrinsics
def print_intrinsics(intrinsics, name=""):
    print(f"{name} Intrinsics:")
    print(f"  Width: {intrinsics.width}")
    print(f"  Height: {intrinsics.height}")
    print(f"  fx: {intrinsics.fx}")
    print(f"  fy: {intrinsics.fy}")
    print(f"  cx: {intrinsics.ppx}")
    print(f"  cy: {intrinsics.ppy}")
    print(f"  Distortion Model: {intrinsics.model}")
    print(f"  Distortion Coefficients: {intrinsics.coeffs}")

# Function to pretty-print extrinsics
def print_extrinsics(extrinsics, name=""):
    print(f"{name} Extrinsics:")
    print(f"  Rotation: {extrinsics.rotation}")
    print(f"  Translation: {extrinsics.translation}")

# Function to pretty-print motion intrinsics
def print_motion_intrinsics(intrinsics, name=""):
    print(f"{name} Motion Intrinsics:")
    print("  Noise Variances:")
    print(f"    {intrinsics.data[0][0]:.6f}, {intrinsics.data[1][1]:.6f}, {intrinsics.data[2][2]:.6f}")
    print("  Bias Variances:")
    print(f"    {intrinsics.data[0][3]:.6f}, {intrinsics.data[1][3]:.6f}, {intrinsics.data[2][3]:.6f}")

# Start the pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable all streams: color, depth, accel, and gyro
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.gyro)

# Start streaming
pipeline_profile = pipeline.start(config)

# Get the active profile and streams
color_profile = pipeline_profile.get_stream(rs.stream.color)
depth_profile = pipeline_profile.get_stream(rs.stream.depth)
gyro_profile = pipeline_profile.get_stream(rs.stream.gyro)
accel_profile = pipeline_profile.get_stream(rs.stream.accel)

# Get intrinsics for color and depth cameras
color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

# Print the intrinsics for the RGB and depth cameras
print_intrinsics(color_intrinsics, "Color")
print_intrinsics(depth_intrinsics, "Depth")

# Get and print extrinsics between the depth and color sensors
depth_to_color_extrinsics = depth_profile.get_extrinsics_to(color_profile)
print_extrinsics(depth_to_color_extrinsics, "Depth to Color")

# Get motion intrinsics (gyro and accelerometer)
gyro_intrinsics = gyro_profile.as_motion_stream_profile().get_motion_intrinsics()
accel_intrinsics = accel_profile.as_motion_stream_profile().get_motion_intrinsics()

# Print the IMU intrinsics (gyro and accel)
print_motion_intrinsics(gyro_intrinsics, "Gyroscope")
print_motion_intrinsics(accel_intrinsics, "Accelerometer")

# Stop the pipeline
pipeline.stop()
