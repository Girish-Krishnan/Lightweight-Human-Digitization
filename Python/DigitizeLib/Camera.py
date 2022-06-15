"""
IMPORTS
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

class Camera:

    def __init__(self,img_size,focal_length,img_center,rotation,translation):
        self.rotation = np.array(rotation)
        self.translation = np.array(translation)
        self.img_id = ""         
        self.intrinsic_matrix = np.array([[0,focal_length[0],-img_center[0]],[-focal_length[1],0,-img_center[1]],[0,0,-1]])
        self.inverse_matrix = np.linalg.inv(self.intrinsic_matrix)
      

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
        self.pcd = np.hstack((np.transpose(np.nonzero(self.depth_map)),np.reshape(self.depth_map[np.nonzero(self.depth_map)],(-1,1))))         
        self.pcd = np.matmul(self.inverse_matrix,np.vstack((self.pcd.T[:2]*self.pcd.T[2],self.pcd.T[2])))[:3].T
        self.colors = np.flip(self.image[np.nonzero(self.depth_map)],axis=1)

    def translate_point_cloud(self,vector):
        self.pcd += vector

    def rotate_point_cloud(self,rotate):    
        self.pcd = np.matmul(rotate,self.pcd.T).T

    def visualize(self):
        self.pcd_o3d = o3d.geometry.PointCloud()
        self.pcd_o3d.points = o3d.utility.Vector3dVector(self.pcd)
        self.pcd_o3d.colors = o3d.utility.Vector3dVector(self.colors/255)
        points = np.asarray(self.pcd_o3d.points)
        self.pcd_o3d = self.pcd_o3d.select_by_index(np.where(points[:,0] > -1000)[0])
        o3d.visualization.draw_geometries([self.pcd_o3d])


class Combiner:

    def __init__(self,cam_array):
        self.cam_array = cam_array

    def combine(self):
        rotate = []
        for i in range(len(self.cam_array)):
            rotate.append(self.cam_array[i].rotation)
            self.cam_array[i].rotate_point_cloud(rotate[i])
            self.cam_array[i].translate_point_cloud(-self.cam_array[i].translation)

        self.pcd = np.concatenate(tuple([i.pcd for i in self.cam_array]),axis=0)
        self.colors = np.concatenate(tuple([i.colors for i in self.cam_array]),axis=0)
        self.complete_pcd = np.hstack((self.pcd,self.colors))
        

    def visualize(self):

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(self.pcd)
        pcd_o3d.colors = o3d.utility.Vector3dVector(self.colors/255)
        points = np.asarray(pcd_o3d.points)
        pcd_o3d = pcd_o3d.select_by_index(np.where(points[:,2] > -1000)[0])
        points = np.asarray(pcd_o3d.points)
        pcd_o3d = pcd_o3d.select_by_index(np.where(points[:,0] > -1000)[0]) 
        o3d.visualization.draw_geometries([pcd_o3d])

        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        viewer.add_geometry(pcd_o3d)
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        viewer.run()
        viewer.destroy_window()
