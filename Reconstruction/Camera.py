"""
IMPORTS
"""
try:
    import cv2 as cv
    import numpy as np
    import open3d as o3d
    import copy
    import pyrealsense2 as rs
    import concurrent.futures
    from tqdm import tqdm
    import os
    import time
    import matplotlib.pyplot as plt
except ImportError as e:
    print("Warning: Unable to import one or more modules due to the following error: ", e)
    pass


"""
Classes for 3D reconstruction
"""

class Camera:

    """
    Initialize a Camera object with the following attributes:
    :param img_size: The size of the image
    :param focal_length: The focal length of the camera
    :param img_center: The center of the image
    :param rotation: The rotation matrix
    :param translation: The translation vector
    """

    def __init__(self,img_size,focal_length,img_center,rotation,translation):
        self.rotation = np.array(rotation)
        self.translation = np.array(translation)
        self.img_id = ""         
        self.fx = focal_length[0]
        self.fy = focal_length[1]
        self.cx = img_center[0]
        self.cy = img_center[1]
        self.img_size = img_size

    def add_image(self,image,depth_map):
        """
        Set the image and the depth map of the Camera object.
        :param image: The image
        :param depth_map: The depth map
        """
        self.image = image
        self.depth_map = depth_map
        
    def display(self):
        """
        Display the image and the depth map (does not need to be called)
        """
        fig = plt.figure()
        fig.add_subplot(1,2,1)
        plt.imshow(cv.cvtColor(self.image,cv.COLOR_BGR2RGB))
        plt.title("Image")
        fig.add_subplot(1,2,2)
        plt.imshow(self.depth_map)
        plt.title("Depth Map")
        plt.show()          

    def point_cloud(self): 
        """
        Convert the Camera object's depth map to a point cloud
        """

        mask = np.logical_or(self.depth_map > 2, self.depth_map < 0.75)
        grads = np.gradient(self.depth_map)
        grad = np.sqrt(grads[0] ** 2 + grads[1] ** 2)

        mask[grad > 0.05] = True

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
        """
        Translate the point cloud of the Camera object
        :param vector: The translation vector

        """
        self.pcd += vector

    def rotate_point_cloud(self,rotate):
        """
        Rotate the point cloud of the Camera object
        :param rotate: The rotation matrix

        """
        self.pcd = np.matmul(rotate,self.pcd.T).T

    def visualize(self):
        """
        Visualize the point cloud of the Camera object
        """

        o3d.visualization.draw_geometries([self.pcd_o3d])


class Combiner:

    """
    Initialize a Combiner object with the following attributes:
    :param cam_array: An array of Camera objects
    """

    def __init__(self,cam_array):
        self.cam_array = cam_array

    def combine(self):
        """
        Combine the point clouds of the Camera objects in the Combiner object
        """

        rotate = []
        for i in range(len(self.cam_array)):
            rotate.append(self.cam_array[i].rotation)
            self.cam_array[i].rotate_point_cloud(rotate[i])
            self.cam_array[i].translate_point_cloud(self.cam_array[i].translation)

        self.pcd = np.concatenate(tuple([i.pcd for i in self.cam_array]),axis=0)
        self.rotate_point_cloud(np.array([[1,0,0],[0,-1,0],[0,0,-1]]))

        self.colors = np.concatenate(tuple([i.colors for i in self.cam_array]),axis=0)
        self.complete_pcd = np.hstack((self.pcd,self.colors))

        self.pcd_o3d = o3d.geometry.PointCloud()
        self.pcd_o3d.points = o3d.utility.Vector3dVector(self.pcd)
        self.pcd_o3d.colors = o3d.utility.Vector3dVector(self.colors)

        self.pcd_o3d, _ = self.pcd_o3d.remove_radius_outlier(1000,radius=0.05)

    def rotate_point_cloud(self,rotate):  
        """
        Rotate the combined point cloud of the Combiner object
        :param rotate: The rotation matrix

        """
        self.pcd = np.matmul(rotate,self.pcd.T).T

    def visualize(self):
        """
        Visualize the combined point cloud of the Combiner object
        """
        o3d.visualization.draw_geometries([self.pcd_o3d])

    def draw_registration_result(self, source, target, transformation):
        """
        Visualize the registration result of the Combiner object (for point-to-point ICP)
        :param source: The source point cloud
        :param target: The target point cloud
        :param transformation: The initial transformation matrix

        """
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
        """
        Apply Point-to-Point ICP to the combined point cloud of the Combiner object
        """
        for i in range(2,len(self.cam_array)):
            source = self.cam_array[i].pcd_o3d
            target = self.cam_array[i-1].pcd_o3d
            threshold = 0.02

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

    """
    
    Initialize a RealSenseCamera object with the following attributes:
    :param serial_number: The serial number of the RealSense camera
    
    """

    def __init__(self, serial_number,width,height,fps) -> None:
        # Create RealSense D415 camera object and pipeline
        self.serial_number = serial_number
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial_number)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.infrared, width, height, rs.format.y8, fps)

        # Start streaming
        self.pipeline.start(self.config)

        # Enable IR emitter and set max laser power
        self.profile = self.pipeline.get_active_profile()
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_sensor.set_option(rs.option.emitter_enabled, 1)
        self.depth_sensor.set_option(rs.option.laser_power, 360) # 360 is max laser power
        self.depth_sensor.set_option(rs.option.global_time_enabled, 1)

        # enable sync between multiple cameras
        self.depth_sensor.set_option(rs.option.inter_cam_sync_mode, 1)

    def get_frames(self):
        """
        :return: The frameset of the RealSenseCamera object
        """
        return self.pipeline.wait_for_frames()
        
    def process_frames(self, frames):
        """
        
        :param frames: The frameset of the RealSenseCamera object

        """    
        aligned_frames = rs.align(rs.stream.depth).process(frames)

        # Get aligned frames
        self.aligned_depth_frame = aligned_frames.get_depth_frame()
        self.aligned_depth_frame = rs.decimation_filter(1).process(self.aligned_depth_frame)
        self.aligned_depth_frame = rs.disparity_transform(True).process(self.aligned_depth_frame)
        self.aligned_depth_frame = rs.spatial_filter().process(self.aligned_depth_frame)
        self.aligned_depth_frame = rs.temporal_filter().process(self.aligned_depth_frame)
        self.aligned_depth_frame = rs.disparity_transform(False).process(self.aligned_depth_frame)

        self.color_frame = aligned_frames.get_color_frame()
        self.raw_color_frame = frames.get_color_frame()

    def save_frames(self, root_directory, sub_directory, numbered=False):
        """
        Save the frames of the RealSenseCamera object as .jpg and .npy files
        """
        # Convert images to numpy arrays
        depth_image = np.asanyarray(self.aligned_depth_frame.get_data())
        color_image = np.asanyarray(self.color_frame.get_data())
        raw_color_image = np.asanyarray(self.raw_color_frame.get_data())
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)

        # Save color as .jpg and depth as .npy
        if not numbered:
            cv.imwrite(root_directory + self.serial_number + "/" + sub_directory + "/image.jpg", color_image)
            cv.imwrite(root_directory + self.serial_number + "/" + sub_directory + "/raw_image.jpg", raw_color_image)
            np.save(root_directory + self.serial_number + "/" + sub_directory + "/depth_map.npy", depth_image)
            cv.imwrite(root_directory + self.serial_number + "/" + sub_directory + "/depth.png", depth_colormap)

        else:
            # Find the number of files in the directory
            num_files = len([f for f in os.listdir(root_directory + self.serial_number + "/" + sub_directory) if os.path.isfile(os.path.join(root_directory + self.serial_number + "/" + sub_directory, f))])
            cv.imwrite(root_directory + self.serial_number + "/" + sub_directory + "/image_" + str(num_files//4) + ".jpg", color_image)
            cv.imwrite(root_directory + self.serial_number + "/" + sub_directory + "/raw_image_" + str(num_files//4) + ".jpg", raw_color_image)
            np.save(root_directory + self.serial_number + "/" + sub_directory + "/depth_map_" + str(num_files//4) + ".npy", depth_image)
            cv.imwrite(root_directory + self.serial_number + "/" + sub_directory + "/depth_" + str(num_files//4) + ".png", depth_colormap)


class SynchronousCapture:
    """
    Initialize a SynchronousCapture object with the following attributes:
    :param serial_numbers: A list of serial numbers of the RealSense cameras
    :param width: The width of the image
    :param height: The height of the image
    :param fps: The frames per second of the image
    :param warmup_time: The time to wait for the cameras to stabilize, in seconds (default 120)
    """

    def __init__(self, serial_numbers,width,height,fps,output_dir,sub_dir,warmup_time=120,numbered=False) -> None:
        self.serial_numbers = serial_numbers
        
        self.cameras = [RealSenseCamera(serial_number,width,height,fps) for serial_number in serial_numbers]
        self.output_dir = output_dir
        self.sub_dir = sub_dir
        self.numbered = numbered
        
        print("[RealSense] Warming up cameras and stabilizing streams. This will take a little more than" + str(warmup_time) + " seconds")

        # Wait for some time to allow camera to stabilize and adjust
        self.capture(int(warmup_time*fps),verbose=False,save_captures=False)

        print("[RealSense] Stabilization Completed")

    def capture(self,capture_count,verbose=False,save_captures=False):
        """
        Capture a single frame from all cameras simultaneously
        """

        for j in tqdm(range(capture_count)):
            tqdm.write("Capturing frame " + str(j+1) + " of " + str(capture_count))
            while True:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    start = time.time()
                    futures = [executor.submit(camera.get_frames) for camera in self.cameras]
                    concurrent.futures.wait(futures)
                    results = [future.result() for future in futures]
                    end = time.time()
                    if verbose:
                        print("Time taken to capture frames: " + str(end - start))

                    if all(results):
                        if save_captures:
                            self.save()
                        break

            # Process frames
            [camera.process_frames(result) for camera, result in zip(self.cameras, results)]

    def save(self):
        """
        Save frames from all cameras simultaneously
        """
        [camera.save_frames(root_directory=self.output_dir,sub_directory=self.sub_dir,numbered=self.numbered) for camera in self.cameras]

    def stop(self):
        """
        Stop streaming from all cameras
        """
        [camera.pipeline.stop() for camera in self.cameras]