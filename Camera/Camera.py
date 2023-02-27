"""
IMPORTS
"""
import cv2 as cv
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    pass
import open3d as o3d
import copy
import pyrealsense2 as rs
import concurrent.futures
import time

"""
Classes for 3D reconstruction
"""

class Camera:

    def __init__(self,img_size,focal_length,img_center,rotation,translation):
        self.rotation = np.array(rotation)
        self.translation = np.array(translation)
        self.img_id = ""         
        self.fx = focal_length[0]
        self.fy = focal_length[1]
        self.cx = img_center[0]
        self.cy = img_center[1]

    def add_image(self,image,depth_map):
        self.image = image
        self.depth_map = depth_map
        
    def display(self): # Display the image and the depth map; does not need to be called
        fig = plt.figure()
        fig.add_subplot(1,2,1)
        plt.imshow(cv.cvtColor(self.image,cv.COLOR_BGR2RGB))
        plt.title("Image")
        fig.add_subplot(1,2,2)
        plt.imshow(self.depth_map)
        plt.title("Depth Map")
        plt.show()          

    def point_cloud(self): 

        mask = np.logical_or(self.depth_map > 2, self.depth_map < 0.75)
        grads = np.gradient(self.depth_map)
        grad = np.sqrt(grads[0] ** 2 + grads[1] ** 2)

        mask[grad > 0.05] = True

        erode_mask = cv.dilate(mask.astype(np.uint8), np.ones((7,7), dtype=np.uint8))

        self.depth_map[mask] = 0

        self.pcd = np.hstack(
            (np.transpose(np.nonzero(self.depth_map)), np.reshape(self.depth_map[np.nonzero(self.depth_map)], (-1,1)) )
        )  # (xxx, 3)
        self.pcd[:, [0, 1]] = self.pcd[:, [1, 0]]  # swap x and y axis since they are reversed in image coordinates

        self.pcd[:, 0] = (self.pcd[:, 0] - self.cx) * self.pcd[:, 2] / self.fx
        self.pcd[:, 1] = (self.pcd[:, 1] - self.cy) * self.pcd[:, 2] / self.fy

        self.colors = np.flip(self.image[np.nonzero(self.depth_map)], axis=1)

        self.pcd_o3d = o3d.geometry.PointCloud()
        self.pcd_o3d.points = o3d.utility.Vector3dVector(self.pcd)
        self.pcd_o3d.colors = o3d.utility.Vector3dVector(self.colors/255)

        self.pcd_o3d, _ = self.pcd_o3d.remove_radius_outlier(1000,radius=0.05)

        self.pcd = np.asarray(self.pcd_o3d.points)
        self.colors = np.asarray(self.pcd_o3d.colors)



    def translate_point_cloud(self,vector):
        self.pcd += vector

    def rotate_point_cloud(self,rotate):    
        self.pcd = np.matmul(rotate,self.pcd.T).T

    def visualize(self):
        o3d.visualization.draw_geometries([self.pcd_o3d])


class Combiner:

    def __init__(self,cam_array):
        self.cam_array = cam_array

    def combine(self):
        rotate = []
        for i in range(len(self.cam_array)):
            rotate.append(self.cam_array[i].rotation)
            self.cam_array[i].rotate_point_cloud(rotate[i])
            self.cam_array[i].translate_point_cloud(self.cam_array[i].translation)

        self.pcd = np.concatenate(tuple([i.pcd for i in self.cam_array]),axis=0)
        #self.pcd = np.matmul(rotate,self.pcd.T).T
        self.rotate_point_cloud(np.array([[1,0,0],[0,-1,0],[0,0,-1]]))

        self.colors = np.concatenate(tuple([i.colors for i in self.cam_array]),axis=0)
        self.complete_pcd = np.hstack((self.pcd,self.colors))

        self.pcd_o3d = o3d.geometry.PointCloud()
        self.pcd_o3d.points = o3d.utility.Vector3dVector(self.pcd)
        self.pcd_o3d.colors = o3d.utility.Vector3dVector(self.colors)

        self.pcd_o3d, _ = self.pcd_o3d.remove_radius_outlier(1000,radius=0.05)

        #self.pcd_o3d.normals = o3d.utility.Vector3dVector(np.zeros((1,3)))

        #self.pcd_o3d.estimate_normals()
        #self.pcd_o3d.orient_normals_consistent_tangent_plane(100)

        #with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        #    self.pcd_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.pcd_o3d, depth=15)

    def rotate_point_cloud(self,rotate):    
        self.pcd = np.matmul(rotate,self.pcd.T).T

    def visualize(self):
        o3d.visualization.draw_geometries([self.pcd_o3d])

    def draw_registration_result(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp],
                                        zoom=0.4459,
                                        front=[0.9288, -0.2951, -0.2242],
                                        lookat=[1.6784, 2.0612, 1.4451],
                                        up=[-0.3402, -0.9189, -0.1996])

    def optimize(self):
        for i in range(2,len(self.cam_array)):
            source = self.cam_array[i].pcd_o3d
            target = self.cam_array[i-1].pcd_o3d
            threshold = 0.02

            # perform ICP

            # trans_init = np.hstack((np.matmul(np.linalg.inv(self.cam_array[i-1].rotation),self.cam_array[i].rotation),
            #                        (-self.cam_array[i-1].translation+self.cam_array[i].translation).reshape(3,1)))
            
            
            #trans_init = np.vstack((trans_init,np.array([0,0,0,1])))

            trans_init = np.eye(4)

            self.draw_registration_result(source, target, trans_init)
            print("Initial alignment")
            evaluation = o3d.pipelines.registration.evaluate_registration(
                 source, target, threshold, trans_init)
            print(evaluation)

            print("Apply point-to-point ICP")
            reg_p2p = o3d.pipelines.registration.registration_icp(
                 source, target, threshold, trans_init,
                 o3d.pipelines.registration.TransformationEstimationPointToPoint())
            print(reg_p2p)
            print("Transformation is:")
            print(reg_p2p.transformation)
            self.draw_registration_result(source, target, reg_p2p.transformation)

            self.cam_array[i].rotation = np.matmul(self.cam_array[i].rotation,np.array(reg_p2p.transformation[:3,:3]))
            self.cam_array[i].translation += np.array(reg_p2p.transformation[:3,3])

"""
Classes for scalable capturing
"""

class RealSenseCamera:

    def __init__(self, serial_number) -> None:
        # Create RealSense D415 camera object and pipeline

        self.serial_number = serial_number
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial_number)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        self.config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 60)

        # Start streaming
        self.pipeline.start(self.config)

        # Enable IR emitter and set max laser power

        self.profile = self.pipeline.get_active_profile()
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_sensor.set_option(rs.option.emitter_enabled, 1)
        self.depth_sensor.set_option(rs.option.enable_auto_exposure, True)
        self.depth_sensor.set_option(rs.option.laser_power, 360)

    def get_frames(self):
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()
        aligned_frames = rs.align(rs.stream.depth).process(frames)

        # Get aligned frames
        self.aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        self.aligned_depth_frame = rs.decimation_filter(1).process(self.aligned_depth_frame)
        self.aligned_depth_frame = rs.disparity_transform(True).process(self.aligned_depth_frame)
        self.aligned_depth_frame = rs.spatial_filter().process(self.aligned_depth_frame)
        self.aligned_depth_frame = rs.temporal_filter().process(self.aligned_depth_frame)
        self.aligned_depth_frame = rs.disparity_transform(False).process(self.aligned_depth_frame)

        self.color_frame = aligned_frames.get_color_frame()
        self.raw_color_frame = frames.get_color_frame()

        # Validate that both frames are valid
        return (self.aligned_depth_frame and self.color_frame)

    def save_frames(self):    
        # Convert images to numpy arrays
        depth_image = np.asanyarray(self.aligned_depth_frame.get_data())
        color_image = np.asanyarray(self.color_frame.get_data())
        raw_color_image = np.asanyarray(self.raw_color_frame.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)

        # Save color as .jpg and depth as .npy
        cv.imwrite(self.serial_number + "/sample_images/image.jpg", color_image)
        cv.imwrite(self.serial_number + "/sample_images/raw_image.jpg", raw_color_image)
        np.save(self.serial_number + "/sample_images/depth_map.npy", depth_image)
        cv.imwrite(self.serial_number + "/sample_images/depth.png", depth_colormap)


class SynchronousCapture:

    def __init__(self, serial_numbers) -> None:
        self.serial_numbers = serial_numbers
        self.cameras = [RealSenseCamera(serial_number) for serial_number in serial_numbers]

    def capture(self):
        # Wait for some time to allow camera to stabilize and adjust
        time.sleep(5)

        # Use multi-threading to capture frames from all cameras simultaneously
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(camera.get_frames) for camera in self.cameras]
            concurrent.futures.wait(futures)

            results = [future.result() for future in futures]
            if not all(results):
                raise Exception("Failed to capture frames from all cameras")

        # Save frames from all cameras
        [camera.save_frames() for camera in self.cameras]

    def stop(self):
        [camera.pipeline.stop() for camera in self.cameras]