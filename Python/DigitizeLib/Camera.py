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
            self.rotation = (param["cam_"+str(self.index)+"_r"]["rotation"])
            self.translation = (param["cam_"+str(self.index)+"_r"]["translation"])
            self.img_id = ""
            self.rgbd = []
            self.intrinsic_matrix = np.array([[self.focal_length[0],0,0],[0,self.focal_length[1],0],[0,0,1]])
            self.rot_and_trans = self.rotation.copy()
            for i in range(len(self.rot_and_trans)):
                self.rot_and_trans[i].append(self.translation[i]) 

            self.rot_and_trans = np.array(self.rot_and_trans)
            self.main_matrix = np.matmul(self.intrinsic_matrix,self.rot_and_trans)

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
        #color = './data/THuman/captures_1024_1024/'+self.img_id+'.png'
        #depth = './data/Depth_Maps/'+ self.img_id +"_depth.png"

        #color_raw = o3d.io.read_image(color)
        #depth_raw = o3d.io.read_image(depth)
        #self.rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

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

        self.rgbd = np.array(self.rgbd)
        xyz=(self.rgbd)[:,:,3:]
        uv = np.copy(xyz)

        for i in uv:
            for j in i:
                j[2] = 1

        rgb=(self.rgbd)[:,:,:3]
        
        xyz_flattened = []

        for i in xyz:
            for j in i:
                xyz_flattened.append(j)

        self.xyz_flattened = np.array(xyz_flattened)

        final = []

        for i in range(len(uv)):
            final.append([])
            for j in range(len(uv[i])):
                final[i].append([])
                
                solution = np.matmul(np.linalg.pinv(self.main_matrix),np.transpose(uv[i][j]))

                print(solution)
                
                if (solution[3] != 0):
                    z = 1/(solution[3])
                    x = z*solution[0]
                    y = z*solution[1]
                
                else:
                    x = 0
                    y = 0
                    z = 0
                
                final[i][j] = [x,y,z]

         
        final_flattened = []

        for i in final:
            for j in i:
                final_flattened.append(j)

        self.final_flattened = np.array(final_flattened)
        

    def visualize(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.xyz_flattened)
        o3d.io.write_point_cloud("./point_cloud.ply", pcd)
        o3d.visualization.draw_geometries([pcd])
