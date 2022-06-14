import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

cfg = pipeline.start()
profile = cfg.get_stream(rs.stream.depth)
intr = profile.as_video_stream_profile().get_intrinsics()
print(intr)