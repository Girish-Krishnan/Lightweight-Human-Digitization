import json
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

class Camera:

    def __init__(self, index):
        
        self.index = index
        param = json.load(open('./data/THuman/thuman_settings2.json'))

        if self.index not in [0,1,2,3]:
            print("Error: Camera "+str(self.index)+" not found!")
        else:
            self.img_size = param["cam_"+str(self.index)+"_r"]["img_size"]
            self.focal_length = param["cam_"+str(self.index)+"_r"]["focal_length"]
            self.img_center = param["cam_"+str(self.index)+"_r"]["img_center"]
            self.rotation = np.array(param["cam_"+str(self.index)+"_r"]["rotation"])
            self.translation = (param["cam_"+str(self.index)+"_r"]["translation"])
            self.img_id = ""
            self.rgbd = []
            self.intrinsic_matrix = np.array([[self.focal_length[0],0,self.img_center[0]],[0,self.focal_length[1],self.img_center[1]],[0,0,1]])
            self.identity_and_translation = np.array([[1,0,0,-self.translation[0]],[0,1,0,-self.translation[1]],[0,0,1,0]])
            self.camera_matrix = np.matmul(np.matmul(self.intrinsic_matrix,self.rotation),self.identity_and_translation)
            self.inverse_matrix = np.linalg.pinv(self.camera_matrix)

    def add_image(self,img_name):
        
        self.img_id = str(img_name) + "_0" + str(self.index)
        self.image = cv.imread('./data/THuman/captures_1024_1024/'+self.img_id+'.png')
        self.depth_map = np.load('./data/THuman/captures_1024_1024/'+self.img_id+'.npy')


    def display(self):
        fig = plt.figure()
        fig.add_subplot(1,2,1)
        plt.imshow(self.image)
        fig.add_subplot(1,2,2)
        plt.imshow(self.depth_map)
        plt.show()

    def get_RGBD(self):
        for i in range(len(self.depth_map)):
             (self.rgbd).append([])
             for j in range(len(self.depth_map[i])):
                    (self.rgbd)[i].append([])
                    (self.rgbd)[i][j].append(i)
                    (self.rgbd)[i][j].append(j)
                    (self.rgbd)[i][j].append(self.depth_map[i][j])
                    (self.rgbd)[i][j].append(self.image[i][j][2])
                    (self.rgbd)[i][j].append(self.image[i][j][1])
                    (self.rgbd)[i][j].append(self.image[i][j][0])

        self.rgbd = np.array(self.rgbd)            

    def point_cloud(self):
        self.pcd = []
        self.colors = []
        for i in range(len(self.depth_map)):
            for j in range(len(self.depth_map[i])):
                if self.depth_map[i][j] != 0:
                    self.pcd.append([i,j,self.depth_map[i][j]])
                    self.colors.append([self.rgbd[i][j][3],self.rgbd[i][j][4],self.rgbd[i][j][5]])

        self.pcd = np.array(self.pcd)
        self.colors = np.array(self.colors)
        
        for i in range(len(self.pcd)):
            self.pcd[i] = np.matmul(self.inverse_matrix,np.hstack((self.pcd[i][:2]*self.pcd[i][2],self.pcd[i][2])))[:3]

    def translate_point_cloud(self,vector):
        for i in range(len(self.pcd)):
            self.pcd[i] = np.add(self.pcd,vector)

    def rotate_point_cloud(self,rotate):    
        for i in range(len(self.pcd)):
            self.pcd[i] = np.matmul(rotate,self.pcd[i])


    def visualize(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pcd)
        pcd.colors = o3d.utility.Vector3dVector(self.colors/255)
        o3d.io.write_point_cloud("./point_cloud.ply", pcd)
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
        rotate_0 = np.array([[0,1,0],[-1,0,0],[0,0,1]])
        rotate_1 = np.array([[1,0,0],[0,0,-1],[0,-1,0]])
        rotate_2 = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        rotate_3 = np.array([[1,0,0],[0,0,1],[0,1,0]])
        
        self.c0.get_RGBD()
        self.c0.point_cloud()
        self.c0.rotate_point_cloud(rotate_0)

        self.c1.get_RGBD()
        self.c1.point_cloud()
        self.c1.rotate_point_cloud(rotate_1)

        self.c2.get_RGBD()
        self.c2.point_cloud()
        self.c2.rotate_point_cloud(rotate_2)

        self.c3.get_RGBD()
        self.c3.point_cloud()
        self.c3.rotate_point_cloud(rotate_3)
        
        pcd_0 = o3d.geometry.PointCloud()
        pcd_0.points = o3d.utility.Vector3dVector(self.c0.pcd)
        pcd_0.colors = o3d.utility.Vector3dVector(self.c0.colors/255)

        pcd_1 = o3d.geometry.PointCloud()
        pcd_1.points = o3d.utility.Vector3dVector(self.c1.pcd)
        pcd_1.colors = o3d.utility.Vector3dVector(self.c1.colors/255)

        pcd_2 = o3d.geometry.PointCloud()
        pcd_2.points = o3d.utility.Vector3dVector(self.c2.pcd)
        pcd_2.colors = o3d.utility.Vector3dVector(self.c2.colors/255)

        pcd_3 = o3d.geometry.PointCloud()
        pcd_3.points = o3d.utility.Vector3dVector(self.c3.pcd)
        pcd_3.colors = o3d.utility.Vector3dVector(self.c3.colors/255)

        o3d.visualization.draw_geometries([pcd_0+pcd_1+pcd_2+pcd_3])