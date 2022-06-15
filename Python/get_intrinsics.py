import pyrealsense2 as rs

pipeline_D435 = rs.pipeline()
config_D435 = rs.config()
config_D435.enable_device('819312073170')
config_D435.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config_D435.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipeline_D415 = rs.pipeline()
config_D415 = rs.config()
config_D415.enable_device('828612060381')
config_D415.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config_D415.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

cfg_D435 = pipeline_D435.start(config_D435)
cfg_D415 = pipeline_D415.start(config_D415)

profile_D415 = cfg_D415.get_stream(rs.stream.depth)
profile_D435 = cfg_D435.get_stream(rs.stream.depth)

intr_D415 = profile_D415.as_video_stream_profile().get_intrinsics()
intr_D435 = profile_D435.as_video_stream_profile().get_intrinsics()
print(intr_D415)
print(intr_D435)