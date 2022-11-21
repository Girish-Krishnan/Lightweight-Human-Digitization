"""
IMPORTS
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import copy

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
        self.pcd = np.hstack(
            (np.transpose(np.nonzero(self.depth_map)), np.reshape(self.depth_map[np.nonzero(self.depth_map)], (-1,1)) )
        )  # (xxx, 3)
        self.pcd[:, [0, 1]] = self.pcd[:, [1, 0]]  # swap x and y axis since they are reversed in image coordinates

        self.pcd[:, 0] = (self.pcd[:, 0] - self.cx) * self.pcd[:, 2] / self.fx
        self.pcd[:, 1] = (self.pcd[:, 1] - self.cy) * self.pcd[:, 2] / self.fy

        self.colors = np.flip(self.image[np.nonzero(self.depth_map)], axis=1)

    def translate_point_cloud(self,vector):
        self.pcd += vector

    def rotate_point_cloud(self,rotate):    
        self.pcd = np.matmul(rotate,self.pcd.T).T

    def visualize(self):
        self.pcd_o3d = o3d.geometry.PointCloud()
        self.pcd_o3d.points = o3d.utility.Vector3dVector(self.pcd)
        self.pcd_o3d.colors = o3d.utility.Vector3dVector(self.colors/255)
        points = np.asarray(self.pcd_o3d.points)
        self.pcd_o3d = self.pcd_o3d.select_by_index(np.where(points[:,2] < 2)[0])
        points = np.asarray(self.pcd_o3d.points)
        self.pcd_o3d = self.pcd_o3d.select_by_index(np.where(points[:,2] > 0.5)[0])
        points = np.asarray(self.pcd_o3d.points)
        self.pcd_o3d = self.pcd_o3d.select_by_index(np.where(points[:,0] > -0.75)[0])
        points = np.asarray(self.pcd_o3d.points)
        self.pcd_o3d = self.pcd_o3d.select_by_index(np.where(points[:,0] < 0.75)[0])
        points = np.asarray(self.pcd_o3d.points)
        self.pcd_o3d = self.pcd_o3d.select_by_index(np.where(points[:,1] < 0.5)[0])
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
        #self.rotate_point_cloud(np.array([[-1,0,0],[0,-1,0],[0,0,-1]]))

        self.colors = np.concatenate(tuple([i.colors for i in self.cam_array]),axis=0)
        self.complete_pcd = np.hstack((self.pcd,self.colors))

        self.pcd_o3d = o3d.geometry.PointCloud()
        self.pcd_o3d.points = o3d.utility.Vector3dVector(self.pcd)
        self.pcd_o3d.colors = o3d.utility.Vector3dVector(self.colors/255)
        
        # cropping
        #points = np.asarray(self.pcd_o3d.points)
        #self.pcd_o3d = self.pcd_o3d.select_by_index(np.where(points[:, 2] < 1)[0])
        #points = np.asarray(self.pcd_o3d.points)
        #self.pcd_o3d = self.pcd_o3d.select_by_index(np.where(points[:, 2] > -2)[0])
        #points = np.asarray(self.pcd_o3d.points)
        #self.pcd_o3d = self.pcd_o3d.select_by_index(np.where(points[:, 0] > -0.75)[0])
        #points = np.asarray(self.pcd_o3d.points)
        #self.pcd_o3d = self.pcd_o3d.select_by_index(np.where(points[:, 0] < 1.5)[0])
        #points = np.asarray(self.pcd_o3d.points)
        #self.pcd_o3d = self.pcd_o3d.select_by_index(np.where(points[:, 1] > -0.4)[0])
        o3d.io.write_point_cloud("./data.ply", self.pcd_o3d)
        
        

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
