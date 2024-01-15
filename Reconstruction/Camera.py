"""
IMPORTS
"""
try:
    import cv2 as cv
    import numpy as np
    import open3d as o3d
    import copy
    import concurrent.futures
    from tqdm import tqdm
    import os
    import time
    from PIL import Image, ImageFilter
    import pyrealsense2 as rs
    
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

    def __init__(self,img_size,focal_length,img_center,rotation,translation,serial_number):
        self.rotation = np.array(rotation)
        self.translation = np.array(translation)
        self.img_id = ""         
        self.fx = focal_length[0]
        self.fy = focal_length[1]
        self.cx = img_center[0]
        self.cy = img_center[1]
        self.img_size = img_size
        self.serial_number = serial_number

    def add_image(self,image,depth_map):
        """
        Set the image and the depth map of the Camera object.
        :param image: The image
        :param depth_map: The depth map
        """
        self.image = image
        self.depth_map = depth_map
        self.mesh = o3d.io.read_triangle_mesh("normal_" + self.serial_number + ".ply")
        self.normals = np.asarray(self.mesh.vertex_normals)
        self.post_process()
        
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

    def post_process(self):
        self.depth_map = crop_depth(self.depth_map,0.1,1.5)
        #self.depth_map = crop_sides(self.depth_map,0.1)
        
        depth_map_data_rgb = Image.fromarray(1000*self.depth_map)
        depth_map_data_rgb = depth_map_data_rgb.convert("RGB")
        depth_map_data_rgb = depth_map_data_rgb.filter(ImageFilter.ModeFilter(size=13))
        depth_map_data_rgb = np.array(depth_map_data_rgb)
        depth_map_data_rgb = depth_map_data_rgb[:, :, 0]
        depth_map_data_rgb = remove_noise(depth_map_data_rgb)
        depth_map_data_rgb = remove_border(depth_map_data_rgb, 17)

        window_size = 5
        variance_threshold = 5
        variance = cv.filter2D(depth_map_data_rgb.astype(float)**2, -1, np.ones((window_size, window_size)), borderType=cv.BORDER_REFLECT)
        variance_mask = variance < variance_threshold
        variance_copy = np.copy(depth_map_data_rgb)
        variance_copy[variance_mask] = 255
        depth_map_data_rgb = cv.bitwise_and(variance_copy, depth_map_data_rgb)
        self.depth_map[depth_map_data_rgb == 0] = 0

    def point_cloud(self): 
        """
        Convert the Camera object's depth map to a point cloud
        """

        # mask = np.logical_or(self.depth_map > 2, self.depth_map < 0.75)
        # grads = np.gradient(self.depth_map)
        # grad = np.sqrt(grads[0] ** 2 + grads[1] ** 2)

        # mask[grad > 0.05] = True

        # self.depth_map[mask] = 0

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

        self.pcd_o3d, ind = self.pcd_o3d.remove_radius_outlier(1000,radius=0.05)

        self.pcd = np.asarray(self.pcd_o3d.points)
        self.colors = np.asarray(self.pcd_o3d.colors)

        filtered_normals = np.asarray(self.normals)[ind]

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.pcd)
        mesh.vertex_colors = o3d.utility.Vector3dVector(self.colors)
        mesh.vertex_normals = o3d.utility.Vector3dVector(filtered_normals)

        self.mesh = mesh
        self.filtered_normals = filtered_normals


    def translate_point_cloud(self,vector):
        """
        Translate the point cloud of the Camera object
        :param vector: The translation vector

        """
        self.pcd += vector
        self.filtered_normals += vector

    def rotate_point_cloud(self,rotate):
        """
        Rotate the point cloud of the Camera object
        :param rotate: The rotation matrix

        """
        self.pcd = np.matmul(rotate,self.pcd.T).T
        self.filtered_normals = np.matmul(rotate,self.filtered_normals.T).T

    def visualize(self):
        """
        Visualize the point cloud of the Camera object
        """

        o3d.visualization.draw_geometries([self.pcd_o3d])


"""
Functions for post-processing
"""

def dynamic_thickness(y, max_thickness, height, min_y, max_y):
    # This function returns a thickness based on y-coordinate
    # You can modify this function to suit your specific requirements
    return max_thickness if y > min_y + (max_y - min_y) / 25 else max_thickness // 3

def draw_contour_with_dynamic_thickness(img, contour, max_thickness):
    height, _, = img.shape
    min_y = np.min(contour[:, :, 1])
    max_y = np.max(contour[:, :, 1])
    for i in range(len(contour) - 1):
        y_value = contour[i][0][1]
        thickness = dynamic_thickness(y_value, max_thickness, height, min_y, max_y)
        
        # Drawing a line segment between two subsequent points
        cv.line(img, tuple(contour[i][0]), tuple(contour[i+1][0]), (0, 0, 0), thickness)
    
    # Closing the contour (connecting the last and the first point)
    y_value = contour[-1][0][1]
    thickness = dynamic_thickness(y_value, max_thickness, height, min_y, max_y)
    cv.line(img, tuple(contour[-1][0]), tuple(contour[0][0]), (0, 0, 0), thickness)
    return img

def remove_border(image, border_thickness):
    # Threshold the grayscale image to create a binary mask
    _, binary_mask = cv.threshold(image, 127, 255, cv.THRESH_BINARY)

    # Find the contours of the human subject
    contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return image
    
    largest_contour = max(contours, key=cv.contourArea)

    # Draw the contour with a thickness to create a border
    #cv.drawContours(binary_mask, contours, -1, (0, 0, 0), border_thickness)
    if len(contours) > 0:
       binary_mask = draw_contour_with_dynamic_thickness(binary_mask, largest_contour, border_thickness)

    return binary_mask

def crop_depth(depth,lower_bound,upper_bound):
    depth[depth < lower_bound] = 0
    depth[depth > upper_bound] = 0
    return depth

def crop_sides(img,percent):
    height, width = img.shape
    img[:,0:int(height*percent)] = 0
    img[:,int(height*(1-percent)):] = 0
    return img

def remove_noise(img):
    _, binary_mask = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return img
    largest_contour = max(contours, key=cv.contourArea)
    output_img = np.zeros_like(img)
    cv.drawContours(output_img, [largest_contour], 0, 255, thickness=cv.FILLED)
    return output_img
    

"""
Classes for 3D reconstruction, combining PCDs
"""

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
        self.normals = np.concatenate(tuple([i.filtered_normals for i in self.cam_array]),axis=0)
        self.rotate_point_cloud(np.array([[1,0,0],[0,-1,0],[0,0,-1]]))

        self.colors = np.concatenate(tuple([i.colors for i in self.cam_array]),axis=0)
        self.complete_pcd = np.hstack((self.pcd,self.colors))

        self.pcd_o3d = o3d.geometry.PointCloud()
        self.pcd_o3d.points = o3d.utility.Vector3dVector(self.pcd)
        self.pcd_o3d.colors = o3d.utility.Vector3dVector(self.colors)
        self.pcd_o3d.normals = o3d.utility.Vector3dVector(self.normals)

        # self.pcd_o3d, ind = self.pcd_o3d.remove_radius_outlier(1000,radius=0.1)

        # Make a mesh using the self.normals
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.pcd)
        mesh.vertex_colors = o3d.utility.Vector3dVector(self.colors)
        mesh.vertex_normals = o3d.utility.Vector3dVector(self.normals)
        mesh.orient_triangles()

        self.mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.pcd_o3d)[0]

        # Poisson Surface Reconstruction Method
        # self.pcd_o3d.estimate_normals()
        # self.pcd_o3d.orient_normals_towards_camera_location()
        # self.mesh_o3d_poisson, self.mesh_id = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.pcd_o3d, depth=8, width=0.0, scale=1.0, linear_fit=False)[0:2]
        # self.mesh_o3d_poisson.compute_vertex_normals()
        # self.mesh_o3d_poisson = self.mesh_o3d_poisson.filter_smooth_simple(number_of_iterations=10)
               
    def rotate_point_cloud(self,rotate):  
        """
        Rotate the combined point cloud of the Combiner object
        :param rotate: The rotation matrix

        """
        self.pcd = np.matmul(rotate,self.pcd.T).T
        self.normals = np.matmul(rotate,self.normals.T).T

    def visualize(self):
        """
        Visualize the combined point cloud of the Combiner object
        """
        o3d.visualization.draw_geometries([self.pcd_o3d])

        # Estimate normals for the point cloud, and then visualize the point cloud with normals
        self.pcd_o3d.estimate_normals()
        points = np.asarray(self.pcd_o3d.points)
        normals = np.asarray(self.pcd_o3d.normals)

        # Create lines for normals from points to (points + normals)
        lines = [[i, i + len(points)] for i in range(len(points))]
        points_with_normals = np.vstack([points, points + 0.02*normals])  # Adjust the 0.02 scalar to scale the length of the normals

        # Create a LineSet from the points and lines
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_with_normals),
            lines=o3d.utility.Vector2iVector(lines),
        )

        # Visualize the point cloud and normals
        o3d.visualization.draw_geometries([self.pcd_o3d, line_set])


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

        # Handle normals
        self.ply = rs.save_to_ply(f"normal_{serial_number}.ply")
        self.ply.set_option(rs.save_to_ply.option_ply_binary, False)
        self.ply.set_option(rs.save_to_ply.option_ply_normals, True)

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
        self.ply.process(aligned_frames)

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
            cv.imwrite(root_directory + '/' + self.serial_number + "/" + sub_directory + "/image.jpg", color_image)
            cv.imwrite(root_directory + '/' + self.serial_number + "/" + sub_directory + "/raw_image.jpg", raw_color_image)
            np.save(root_directory + '/' + self.serial_number + "/" + sub_directory + "/depth_map.npy", depth_image)
            cv.imwrite(root_directory + '/' + self.serial_number + "/" + sub_directory + "/depth.png", depth_colormap)

        else:
            # Find the number of files in the directory
            num_files = len([f for f in os.listdir(root_directory + "/" + self.serial_number + "/" + sub_directory) if os.path.isfile(os.path.join(root_directory + self.serial_number + "/" + sub_directory, f))])
            cv.imwrite(root_directory + '/' + self.serial_number + "/" + sub_directory + "/image_" + str(num_files//4) + ".jpg", color_image)
            cv.imwrite(root_directory + '/' + self.serial_number + "/" + sub_directory + "/raw_image_" + str(num_files//4) + ".jpg", raw_color_image)
            np.save(root_directory + '/' + self.serial_number + "/" + sub_directory + "/depth_map_" + str(num_files//4) + ".npy", depth_image)
            cv.imwrite(root_directory + '/' + self.serial_number + "/" + sub_directory + "/depth_" + str(num_files//4) + ".png", depth_colormap)


class SynchronousCapture:
    """
    Initialize a SynchronousCapture object with the following attributes:
    :param serial_numbers: A list of serial numbers of the RealSense cameras
    :param width: The width of the image
    :param height: The height of the image
    :param fps: The frames per second of the image
    :param warmup_frames: The number of frames to wait for the cameras to stabilize (default 1000)
    """

    def __init__(self, serial_numbers,width,height,fps,output_dir,sub_dir,warmup_frames=1000,numbered=False) -> None:
        self.serial_numbers = serial_numbers
        
        self.cameras = [RealSenseCamera(serial_number,width,height,fps) for serial_number in serial_numbers]
        self.output_dir = output_dir
        self.sub_dir = sub_dir
        self.numbered = numbered
        
        print("[RealSense] Warming up cameras and stabilizing streams")

        # Wait for some time to allow camera to stabilize and adjust
        self.capture(int(warmup_frames),verbose=False,save_captures=False, process_frames=False)

        print("[RealSense] Stabilization Completed")

    def capture(self,capture_count,verbose=False,save_captures=False, process_frames=True):
        """
        Capture a single frame from all cameras simultaneously
        """

        for j in tqdm(range(capture_count)):
            while True:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    start = time.time()
                    futures = [executor.submit(camera.get_frames) for camera in self.cameras]
                    concurrent.futures.wait(futures)
                    results = [future.result() for future in futures]
                    end = time.time()
                    if verbose:
                        print("Time taken to capture frames: " + str(end - start))

                if process_frames:
                    [camera.process_frames(result) for camera, result in zip(self.cameras, results)]

                if all(results):
                    if save_captures:
                        self.save()
                
                break

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