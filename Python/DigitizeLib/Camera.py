"""
IMPORTS
"""
import json
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

"""
CONSTANTS
"""

NUM_CAMS = 4 # Number of cameras
TRANSITION_MATRIX = np.array([[0,-1,0],[1,0,0],[0,0,-1]]) # transition matrix


param = json.load(open('./data/THuman/thuman_settings2.json'))

class Camera:

    def __init__(self, index):
        
        self.index = index

        if self.index not in range(NUM_CAMS):
            print("Error: Camera "+str(self.index)+" not found!")
        else:
            self.img_size = np.array(param["cam_"+str(self.index)+"_r"]["img_size"])
            self.focal_length = np.array(param["cam_"+str(self.index)+"_r"]["focal_length"])
            self.img_center = np.array(param["cam_"+str(self.index)+"_r"]["img_center"])
            self.rotation = np.array(param["cam_"+str(self.index)+"_r"]["rotation"])
            self.translation = np.array(param["cam_"+str(self.index)+"_r"]["translation"])
            self.img_id = ""
            
            self.intrinsic_matrix = np.array([[self.focal_length[0],0,self.img_center[0]],[0,self.focal_length[1],self.img_center[1]],[0,0,1]])
            self.identity = np.hstack((np.eye(3),np.zeros((3,1))))
            self.camera_matrix = np.matmul(self.intrinsic_matrix,self.identity)
            self.inverse_matrix = np.linalg.pinv(self.camera_matrix)

    def add_image(self,img_name):
        
        self.img_id = str(img_name) + "_0" + str(self.index)
        self.image = cv.imread('./data/THuman/captures_1024_1024/'+self.img_id+'.png')
        self.depth_map = np.load('./data/THuman/captures_1024_1024/'+self.img_id+'.npy')
        
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
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pcd)
        pcd.colors = o3d.utility.Vector3dVector(self.colors/255)
        o3d.visualization.draw_geometries([pcd])


class Combiner:

    def __init__(self):
        self.c0 = Camera(0)
        self.c1 = Camera(1)
        self.c2 = Camera(2)
        self.c3 = Camera(3)


    def add_image(self,image_name):

        self.c0.add_image(image_name)
        self.c1.add_image(image_name)
        self.c2.add_image(image_name)
        self.c3.add_image(image_name)

    def add(self):
        rotate_0 = np.matmul(self.c0.rotation,TRANSITION_MATRIX)
        rotate_1 = np.matmul(self.c1.rotation,TRANSITION_MATRIX)
        rotate_2 = np.matmul(self.c2.rotation,TRANSITION_MATRIX)
        rotate_3 = np.matmul(self.c3.rotation,TRANSITION_MATRIX)

        self.c0.point_cloud()
        self.c0.rotate_point_cloud(rotate_0)
        self.c0.translate_point_cloud(np.array(-self.c0.translation))

        self.c1.point_cloud()
        self.c1.rotate_point_cloud(rotate_1)
        self.c1.translate_point_cloud(np.array(-self.c1.translation))

        self.c2.point_cloud()
        self.c2.rotate_point_cloud(rotate_2)
        self.c2.translate_point_cloud(np.array(-self.c2.translation))

        self.c3.point_cloud()
        self.c3.rotate_point_cloud(rotate_3)
        self.c3.translate_point_cloud(np.array(-self.c3.translation))

        self.pcd = np.concatenate((self.c0.pcd,self.c1.pcd,self.c2.pcd,self.c3.pcd),axis=0)
        self.colors = np.concatenate((self.c0.colors,self.c1.colors,self.c2.colors,self.c3.colors),axis=0)

        self.complete_pcd = np.hstack((self.pcd,self.colors))

        np.save("./point_cloud.ply",self.complete_pcd)
        

    def visualize(self):

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(self.pcd)
        pcd_o3d.colors = o3d.utility.Vector3dVector(self.colors/255)

        o3d.visualization.draw_geometries([pcd_o3d])