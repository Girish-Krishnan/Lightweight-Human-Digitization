import pyrealsense2 as rs
SERIAL = '828612060381'

config = rs.config()
resolution_width = 480
resolution_height = 270
framerate = 90
config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, framerate)
config.enable_stream(rs.stream.infrared, 2, resolution_width, resolution_height, rs.format.y8, framerate)
config.enable_device(SERIAL)

pipe = rs.pipeline()
prof = pipe.start(config)
dev = prof.get_device()
ds = dev.query_sensors()[0]
ds.set_option(rs.option.inter_cam_sync_mode, 2)  