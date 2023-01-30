import numpy as np
import json
import cv2
import pyrealsense2 as rs
from pykalman import KalmanFilter

"""
GET CAM CALIBRATION DATA
"""

SETTINGS_PATH = './configuration_parameters.json'
param = json.load(open(SETTINGS_PATH))
cams_list = list(param["cams"].keys())
print("cams_list: ", cams_list)

video_paths = ['./'+c+'/video.bag' for c in cams_list]
pipelines = []
configs = []
profiles = []
playbacks = []
video_duration = 0
frame_rate = 60

# Create a Kalman filter
kf = KalmanFilter(transition_matrices=np.eye(480*640),
                  observation_matrices=np.eye(480*640))

for i in range(len(video_paths)):
    pipelines.append(rs.pipeline())
    configs.append(rs.config())
    rs.config.enable_device_from_file(configs[i], video_paths[i])
    configs[i].enable_stream(rs.stream.depth, rs.format.z16, 60)
    configs[i].enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 60)
    profiles.append(pipelines[i].start(configs[i]))
    playbacks.append(profiles[i].get_device().as_playback())
    # playbacks[i].set_real_time(False)
    video_duration = playbacks[i].get_duration().total_seconds()


num_frames = int(video_duration * frame_rate)

print('Number of frames: ',num_frames)

frame_count = 0

# Initialize an empty frame for the filtered depth map
filtered_depth_maps = [np.zeros((480,640))]*len(video_paths)

align = rs.align(rs.stream.depth)

while True:
        for i in range(len(video_paths)):
            # Get frameset of depth
            try:
                frames = pipelines[i].wait_for_frames()
                frames = align.process(frames)
                # Get depth frame
                depth_frame = frames.get_depth_frame()

                depth_frame = rs.decimation_filter(1).process(depth_frame)
                depth_frame = rs.disparity_transform(True).process(depth_frame)
                depth_frame = rs.spatial_filter().process(depth_frame)
                depth_frame = rs.temporal_filter().process(depth_frame)
                depth_frame = rs.disparity_transform(False).process(depth_frame)

                depth_frame = np.asanyarray(depth_frame.get_data())
                
                filtered_depth_maps[i] = depth_frame

                if frame_count > 0:
                    filtered_depth_maps[i], _ = kf.filter_update(filtered_depth_maps[i], depth_frame)

            
            except Exception as e:
                print(e)
                break
            
        frame_count +=1

        if frame_count == num_frames:
            break

print(cv2.applyColorMap(cv2.convertScaleAbs(filtered_depth_maps[0]/num_frames, alpha=0.03), cv2.COLORMAP_JET))

for i in range(len(cams_list)):
    # Save the filtered depth map
    cv2.imwrite("./" + cams_list[i] +  "/sample_images/filtered_depth_map.png", cv2.applyColorMap(cv2.convertScaleAbs(filtered_depth_maps[i]/num_frames, alpha=0.03), cv2.COLORMAP_JET))