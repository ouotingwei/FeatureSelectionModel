import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.linalg import logm
import math

class generate_label:
    def __init__(self, input_array, img, img_size, camera_intrinsic, T):
        self.input_array = input_array
        self.num_of_points = len(input_array)-1
        self.camera_matrix = np.array([[camera_intrinsic[0], 0, camera_intrinsic[2]], [0, camera_intrinsic[1], camera_intrinsic[3]], [0, 0, 1]], dtype=float)
        self.transition = T
        self.h = img_size[0]
        self.img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.w = img_size[1]

    def get_label(self):
        # set inlier && outlier label by T
        error_list, rate = self.find_reprojection_error()

        return error_list, rate
    
    def find_reprojection_error(self):
        K = self.camera_matrix
        error_list = []
        for point in self.input_array:
            point_3d = np.array([point[1][0], point[1][1], point[1][2], 1])
            transformed_point_3d = self.transition @ point_3d  # 4x4 @ 4x1 -> 4x1
            # 取前三个元素，舍去最后一个元素
            transformed_point_3d = transformed_point_3d[:3]
            uvw = K @ transformed_point_3d  # 3x3 @ 3x1 -> 3x1
            u = uvw[0] / uvw[2]  # 归一化
            v = uvw[1] / uvw[2]
            error_list.append(math.sqrt((u - point[0][0]) ** 2 + (v - point[0][1]) ** 2))
        
        zero_error_cnt = 0
        u_list = []
        v_list = []
        u_out = []
        v_out = []
        for i in range(len(error_list)):
            if error_list[i] < 5.991:
                zero_error_cnt += 1
                u_list.append(self.input_array[i][0][0] )
                v_list.append(self.input_array[i][0][1] )

            else:
                u_out.append(self.input_array[i][0][0] )
                v_out.append(self.input_array[i][0][1] )
        
        # 计算中位数和标准差
        error_list_np = np.array(error_list)
        median_error = np.median(error_list_np)
        std_error = np.std(error_list_np)
        
        #print("Error list:", error_list)
        print("Median of error list:", median_error)
        print("Standard deviation of error list:", std_error)
        
        print("reprojection inlier rate = ", len(u_list)/len(error_list)*100, " %")
        
        if (len(u_list)/len(error_list)*100 > 70 and len(u_list)/len(error_list)*100 < 90):
            # show inlier / outlier (uv)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(self.img)
            plt.axis('off')
            plt.scatter(u_list, v_list, color='blue', label='Inliers')
            plt.scatter(u_out, v_out, color='red', label='Outliers')
            plt.legend()
            plt.title('Inliers and Outliers')
            plt.xlabel('u')
            plt.ylabel('v')

            plt.show()
        

        '''
        plt.title("Projection error by the accurate Tcw")
        plt.xlabel("Key points")
        plt.ylabel("Pixel")
        plt.plot( error_list, marker='o', linestyle='' )
        plt.show()
        '''

        return error_list, len(u_list)/len(error_list)*100
