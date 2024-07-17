import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.linalg import logm
import math
import time

import orb_operation as ORB

class Reprojection:
    def __init__(self, point_3d_query, point_2d_train, octave, img, camera_intrinsic, T):
        self.point_3d_query = point_3d_query
        self.point_2d_train = point_2d_train
        self.octave = octave
        self.camera_matrix = np.array([[camera_intrinsic[0], 0, camera_intrinsic[2]], [0, camera_intrinsic[1], camera_intrinsic[3]], [0, 0, 1]], dtype=float)
        self.transition = T
        self.img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    def get_reprojection(self):
        # set inlier && outlier label by T
        error_list, rate = self.find_reprojection_error()

        return error_list, rate
    
    def find_reprojection_error(self):
        K = self.camera_matrix
        error_list = []
        reprojection_points = []
        for i in range(len(self.point_3d_query)):
            point_3d = np.array([self.point_3d_query[i][0], self.point_3d_query[i][1], self.point_3d_query[i][2], 1])
            transformed_point_3d = self.transition @ point_3d  
            transformed_point_3d = transformed_point_3d[:3]
            uvw = K @ transformed_point_3d 
            u = uvw[0] / uvw[2]  
            v = uvw[1] / uvw[2]
            reprojection_points.append([u, v])
            error_list.append(math.sqrt((u - self.point_2d_train[i][0]) ** 2 + (v - self.point_2d_train[i][1]) ** 2))
        
        u_in, v_in, u_out, v_out, depth_zero_u, depth_zero_v = [], [], [], [], [], []
        cnt_in = 0
        cnt_out = 0
        cnt_zero = 0
        for i in range(len(error_list)):
            if self.point_3d_query[i][2] == 0.0:
                cnt_zero +=1
                depth_zero_u.append(self.point_2d_train[i][0])
                depth_zero_v.append(self.point_2d_train[i][1])
            elif error_list[i] < 5.991 * (self.octave[i]+1):
                cnt_in += 1
                u_in.append(self.point_2d_train[i][0])
                v_in.append(self.point_2d_train[i][1])
            else:
                cnt_out += 1
                u_out.append(self.point_2d_train[i][0])
                v_out.append(self.point_2d_train[i][1])
            
        print("haha",len(error_list), cnt_in, cnt_out, cnt_zero )
        
        valid_error_count = len(error_list) - len(depth_zero_u)
        if valid_error_count > 0 and (len(u_in) / valid_error_count * 100) > 99:
            plt.figure(figsize=(10, 6))
            plt.imshow(self.img)
            plt.axis('off')
            plt.scatter(u_in, v_in, color='green', label='Inliers')
            plt.scatter(u_out, v_out, color='red', label='Outliers')
            plt.scatter(depth_zero_u, depth_zero_v, color='blue', label='Ignore')

            for idx in range(len(self.point_2d_train)):
                if self.point_3d_query[idx][2] != 0.0:  # 只绘制深度不为0的点的重投影线
                    u_original = self.point_2d_train[idx][0]
                    v_original = self.point_2d_train[idx][1]
                    u_reprojection = reprojection_points[idx][0]
                    v_reprojection = reprojection_points[idx][1]
                    plt.plot([u_original, u_reprojection], [v_original, v_reprojection], color='yellow')

            plt.legend()
            plt.title('Inliers and Outliers with Reprojection Lines')
            plt.xlabel('u')
            plt.ylabel('v')
            plt.show()

        return error_list, len(u_in) / len(error_list) * 100


class Visibility:
    def __init__(self, color_img_list, now_sequence, now_img, now_kp, now_des, queryIdx, VisibilityInterval) -> None:
        end_sequence = now_sequence + VisibilityInterval
        self.visible_img_list = color_img_list[now_sequence+1:end_sequence]  # 確保區間長度正確
        self.base_kp = [now_kp[i] for i in queryIdx]
        self.base_des = cv.UMat(np.array([now_des[i] for i in queryIdx]))
        self.visibility_interval = VisibilityInterval
        self.now_img = now_img
        self.now_seq = now_sequence

    def get_visibility(self):
        match_list = np.ones(len(self.base_kp))
        visible_list = np.zeros(len(self.base_kp))

        for img_path in self.visible_img_list:  # 確保在當前區間內的範圍
            ne_img = cv.imread(img_path)

            # ORB Feature Extraction
            ne_image_ = ORB.orb_features(ne_img)
            ne_kp, ne_des = ne_image_.feature_extract() 

            # matching the feature between two image frames
            matcher_ = ORB.feature_match(self.now_img, ne_img, self.base_kp, ne_kp, self.base_des, ne_des) 
            quIdx, trIdx = matcher_.frame_match()  # now->query, next->train

            for k in range(len(quIdx)):
                match_list[quIdx[k]] += 1
        cnt = 0

        for i in range(len(match_list)):
            if match_list[i] == self.visibility_interval:
                visible_list[i] = 1
                cnt += 1

        rate = cnt / len(match_list) * 100

        return visible_list, rate
    
class DrawStability():
    def __init__(self, now_img, queryIdx, input_array, error_list, visible_list):
        self.now_img = now_img
        self.input_array = input_array
        self.error_list = error_list
        self.visible_list = visible_list
        self.queryIdx = queryIdx
    
    def draw_stability(self):
        stable_u, stable_v, unstable_u, unstable_v, mismatch_u, mismatch_v = [], [], [], [], [], []
        for i in range(len(self.queryIdx)):
            if self.visible_list[i] == 1 and self.error_list[i] < 1:
                # stable
                pass
            
            if (self.visible_list[i] != 1 and self.error_list[i] < 1) or (self.visible_list[i] == 1 and self.error_list[i] < 5.991 and self.error_list[i] >= 1):
                # unstable
                pass

            if self.error_list[i] > 5.991:
                # mismatch
                pass
        
        plt.figure(figsize=(10, 6))
        plt.imshow(self.img)
        plt.axis('off')
        plt.scatter(stable_u, stable_v, color='green', label='Stable')
        plt.scatter(unstable_u, unstable_v, color='red', label='Unstable')
        plt.scatter(mismatch_u, mismatch_v, color='blue', label='Mismatch')
        plt.legend()
        plt.title('Stability')
        plt.xlabel('u')
        plt.ylabel('v')

        plt.show()
