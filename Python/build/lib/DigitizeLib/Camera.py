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
            self.rotation = param["cam_"+str(self.index)+"_r"]["rotation"]
            self.translation = param["cam_"+str(self.index)+"_r"]["translation"]
            cx = self.img_center[0]
            cy = self.img_center[1]
            S = self.rotation
            
            self.img_id = ""
            self.rgbd = []

    def add_image(self,img_name):
        
        self.img_id = str(img_name) + "_0" + str(self.index)
        self.image = cv.imread('./data/THuman/captures_1024_1024/'+self.img_id+'.png')
        self.depth_map = np.load('./data/THuman/captures_1024_1024/'+self.img_id+'.npy')
        
        # to be removed soon
        cv.imwrite('./data/Depth_Maps/'+ self.img_id +'_depth.png',self.depth_map)


    def display(self):
        fig = plt.figure()
        fig.add_subplot(1,2,1)
        plt.imshow(self.image)
        fig.add_subplot(1,2,2)
        plt.imshow(self.depth_map)
        plt.show()

    def get_RGBD(self):
        color = './data/THuman/captures_1024_1024/'+self.img_id+'.png'
        depth = './data/Depth_Maps/'+ self.img_id +"_depth.png"

        color_raw = o3d.io.read_image(color)
        depth_raw = o3d.io.read_image(depth)
        self.rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

        for i in range(len(self.depth_map)):
             (self.rgbd).append([])
             for j in range(len(self.depth_map[i])):
                 #if (self.depth_map[i][j] != 0):
                    (self.rgbd)[i].append([])
                    (self.rgbd)[i][j].append(i)
                    (self.rgbd)[i][j].append(j)
                    (self.rgbd)[i][j].append(self.depth_map[i][j])
                    (self.rgbd)[i][j].append(self.image[i][j][2])
                    (self.rgbd)[i][j].append(self.image[i][j][1])
                    (self.rgbd)[i][j].append(self.image[i][j][0])
                    

        
        
    def display_RGBD(self):
        plt.subplot(1, 2, 1)
        plt.title('Image')
        plt.imshow(self.rgbd_image.color)
        plt.subplot(1, 2, 2)
        plt.title('Depth image')
        plt.imshow(self.rgbd_image.depth)
        plt.show()

    def point_cloud(self):
        # self.pcd = o3d.geometry.PointCloud.create_from_rgbd_image(self.rgbd_image,o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        # self.pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        #mean_Z=np.mean(self.rgbd,axis=0)[2]
        #spatial_query=self.rgbd[abs(self.rgbd[:,2]-mean_Z)<1]
        xyz=(self.rgbd)[:][:][:][3:]
        rgb=(self.rgbd)[:][:][:][:3]
        
        ax = plt.axes(projection='3d')

        cn = rgb.copy()
        
        for i in cn:
            for j in i:
                for k in j:
                    k = k/255

        print(rgb)
        #ax.scatter(xyz[:][:][0], xyz[:][:][1], xyz[:][:][2], c = cn, s=0.01)
        #plt.show()

        

    #def visualize(self):
    #    o3d.visualization.draw_geometries(self.p_cloud)