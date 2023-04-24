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
    import subprocess
    import itertools
    import time
    import matplotlib.pyplot as plt
except ImportError as e:
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
        #self.pcd = np.matmul(rotate,self.pcd.T).T
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
        #self.depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
        #self.depth_sensor.set_option(rs.option.enable_auto_white_balance, 1)
        #self.depth_sensor.set_option(rs.option.output_trigger_enabled, 1)
        self.depth_sensor.set_option(rs.option.laser_power, 360)
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
        self.aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        self.aligned_depth_frame = rs.decimation_filter(1).process(self.aligned_depth_frame)
        self.aligned_depth_frame = rs.disparity_transform(True).process(self.aligned_depth_frame)
        self.aligned_depth_frame = rs.spatial_filter().process(self.aligned_depth_frame)
        self.aligned_depth_frame = rs.temporal_filter().process(self.aligned_depth_frame)
        self.aligned_depth_frame = rs.disparity_transform(False).process(self.aligned_depth_frame)

        self.color_frame = aligned_frames.get_color_frame()
        self.raw_color_frame = frames.get_color_frame()

    def save_frames(self):
        """
        Save the frames of the RealSenseCamera object as .jpg and .npy files
        """
        # Convert images to numpy arrays
        depth_image = np.asanyarray(self.aligned_depth_frame.get_data())
        color_image = np.asanyarray(self.color_frame.get_data())
        raw_color_image = np.asanyarray(self.raw_color_frame.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)

        # Save color as .jpg and depth as .npy
        cv.imwrite("./Camera_Data/" + self.serial_number + "/sample_images/image.jpg", color_image)
        cv.imwrite("./Camera_Data/" + self.serial_number + "/sample_images/raw_image.jpg", raw_color_image)
        np.save("./Camera_Data/" + self.serial_number + "/sample_images/depth_map.npy", depth_image)
        cv.imwrite("./Camera_Data/" + self.serial_number + "/sample_images/depth.png", depth_colormap)


class SynchronousCapture:
    """
    Initialize a SynchronousCapture object with the following attributes:
    :param serial_numbers: A list of serial numbers of the RealSense cameras
    """

    def __init__(self, serial_numbers,buffer=False) -> None:
        self.serial_numbers = serial_numbers
        
        self.cameras = [RealSenseCamera(serial_number) for serial_number in serial_numbers]
        
        # Wait for some time to allow camera to stabilize and adjust
        print("Waiting for cameras to stabilize...")
        if not buffer:
            self.capture(50,verbose=False,data_collection=False)
        else:
            self.capture_buffer(50,verbose=False,data_collection=False)

        print("Stabilization Completed")

    def capture(self,capture_count,verbose=False,data_collection=True):
        """
        Capture a single frame from all cameras simultaneously
        """
        if data_collection:
            data_file = "./test_data_single_frame.csv"
            file = open(data_file, "w")
            file.write("Experiment, Timestamp Range, Timestamp Mean Difference\n")
            file.close()

            file = open(data_file, "a")

        for j in range(capture_count):
            while True:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    start = time.time()
                    futures = [executor.submit(camera.get_frames) for camera in self.cameras]
                    concurrent.futures.wait(futures)
                    results = [future.result() for future in futures]
                    end = time.time()
                    if verbose:
                        print("#############################################")
                        print("Results")
                        print("#############################################")
                        print("Time taken to capture frames: " + str(end - start))
                        # Print out timestamp of each result frame
                    
                    for i, result in enumerate(results):
                        # timestamp = result.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)
                        timestamp = result.get_timestamp()
                        domain = result.get_frame_timestamp_domain()
                        if verbose:
                            print("Camera ", self.serial_numbers[i], " timestamp: ", timestamp, "domain: ", domain)

                    timestamps = [result.get_timestamp() for result in results]

                    timestamp_range = max(timestamps) - min(timestamps)
                    mean_difference = np.mean(np.abs(np.array(timestamps) - timestamps[0]))

                    if verbose:
                        print("Timestamp range: ", timestamp_range)
                        print("Mean difference: ", mean_difference)
                    
                    if data_collection:
                        file.write(str(j+1) + ", " + str(timestamp_range) + ", " + str(mean_difference) + "\n")

                    if all(results):
                        break

            # Process frames
            [camera.process_frames(result) for camera, result in zip(self.cameras, results)]

        if data_collection:
            file.close()

    def capture_buffer(self, capture_count, verbose=False, data_collection=True):
        """
        Capture a buffer of 3 frames from all cameras simultaneously
        """
        if data_collection:
            data_file = "./test_data_buffer.csv"
            file = open(data_file, "w")
            file.write("Experiment, Timestamp Range, Timestamp Mean Difference\n")
            file.close()

            file = open(data_file, "a")

        for j in range(capture_count):
            while True:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    start = time.time()
                    futures_1 = [executor.submit(camera.get_frames) for camera in self.cameras]
                    concurrent.futures.wait(futures_1)
                    results_1 = [future.result() for future in futures_1]

                    futures_2 = [executor.submit(camera.get_frames) for camera in self.cameras]
                    concurrent.futures.wait(futures_2)
                    results_2 = [future.result() for future in futures_2]

                    futures_3 = [executor.submit(camera.get_frames) for camera in self.cameras]
                    concurrent.futures.wait(futures_3)
                    results_3 = [future.result() for future in futures_3]

                    end = time.time()

                    if not all(results_1) or not all(results_2) or not all(results_3):
                        continue

                    else:
                        if verbose:
                            print("#############################################")
                            print("Results")
                            print("#############################################")
                        # Print out timestamp of each result frame
                        for i, result in enumerate(results_1):
                            # timestamp = result.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)
                            timestamp = result.get_timestamp()
                            domain = result.get_frame_timestamp_domain()
                            if verbose:
                                print("Camera ", self.serial_numbers[i], " timestamp: ", timestamp, "domain: ", domain)

                        for i, result in enumerate(results_2):
                            timestamp = result.get_timestamp()
                            domain = result.get_frame_timestamp_domain()
                            if verbose:
                                print("Camera ", self.serial_numbers[i], " timestamp: ", timestamp, "domain: ", domain)

                        for i, result in enumerate(results_3):
                            timestamp = result.get_timestamp()
                            domain = result.get_frame_timestamp_domain()
                            if verbose:
                                print("Camera ", self.serial_numbers[i], " timestamp: ", timestamp, "domain: ", domain)
                        
                        results = [results_1, results_2, results_3]
                        # Transpose the list of lists, similar to a matrix transpose
                        results = list(map(list, zip(*results)))

                        # Find standard deviation of timestamps
                        timestamps_1 = [result.get_timestamp() for result in results_1]
                        timestamps_2 = [result.get_timestamp() for result in results_2]
                        timestamps_3 = [result.get_timestamp() for result in results_3]

                        timestamps, timestamp_range, mean_diff, frame_indices = self.closest_timestamps(timestamps_1, timestamps_2, timestamps_3)

                        if data_collection:
                            file.write(str(j+1) + ", " + str(timestamp_range) + ", " + str(mean_diff) + "\n")

                        if verbose:
                            print("#### USING A BUFFER OF 3 FRAMES ####")

                            print("Range of timestamps: ", timestamp_range)
                            print("Mean difference between timestamps: ", mean_diff)

                            #print("Time taken to capture frames from all cameras: ", "{:.21f}".format(end - start))
                            print("#############################################")

                            print("Indices of best frames (0, 1, or 2) for each cam")
                            print(frame_indices)

                        best_results = []

                        for i in range(len(self.cameras)):
                            best_results.append(results[i][frame_indices[i]])

                        # Process frames
                        [camera.process_frames(best_results[i]) for i, camera in enumerate(self.cameras)]
                        break

        if data_collection:
            file.close() 
    
    def closest_timestamps(self,*timestamps):

        timestamps = np.array(timestamps)
        timestamps = np.transpose(timestamps)

        center_frame_idx = timestamps.shape[1] // 2
        reference_timestamp = timestamps[0][center_frame_idx]

        frame_indices = []

        for i in range(timestamps.shape[0]):
            frame_indices.append(np.argmin(np.abs(timestamps[i] - reference_timestamp)))
        
        chosen = timestamps[:,frame_indices]
        timestamp_range = np.max(chosen) - np.min(chosen)

        # Find mean difference between timestamps
        mean_diff = np.mean(np.abs(chosen - reference_timestamp))

        return timestamps, timestamp_range, mean_diff, frame_indices

    def closest_timestamps_brute_force(self, *lists):
        """
        Given a list of lists (all lists the same length)
        choose one number from each list such that range of chosen numbers is minimized
        You can for example choose the first element from list 1, the third element from list 2, etc.
        The function returns the chosen numbers and the range of the chosen numbers
        """
        # Create a list of lists of all possible combinations
        all_combinations = list(itertools.product(*lists))
        # Find the combination with the smallest range
        min_range = min([max(combination) - min(combination) for combination in all_combinations])
        # Find the index of the combination with the smallest range
        min_range_index = [max(combination) - min(combination) for combination in all_combinations].index(min_range)

        # Find the indices of the combination from each list
        indices = [list.index(combination) for list, combination in zip(lists, all_combinations[min_range_index])]

        chosen = all_combinations[min_range_index]
        # Find mean difference between timestamps
        mean_diff = np.mean(np.abs(np.array(chosen) - chosen[0]))
        
        # Return the combination with the smallest range, the range value, and the indices of the combination from each list
        return chosen, min_range, mean_diff, indices

    def save(self):
        """
        Save frames from all cameras simultaneously
        """
        [camera.save_frames() for camera in self.cameras]

    def stop(self):
        """
        Stop streaming from all cameras
        """
        [camera.pipeline.stop() for camera in self.cameras]