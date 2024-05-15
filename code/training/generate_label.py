import numpy as np
import time
import cv2 as cv
from itertools import combinations
from tqdm import tqdm
import random
from numba import jit
import math

class generate_label:
    def __init__(self, input_array, img_size, camera_intrinsic):
        self.input_array = input_array
        self.num_of_points = len(input_array)-1
        self.camera_matrix = np.array([[camera_intrinsic[0], 0, camera_intrinsic[2]], [0, camera_intrinsic[1], camera_intrinsic[3]], [0, 0, 1]], dtype=float)
        self.h = img_size[0]
        self.w = img_size[1]
        self.keep_ratio = 0.2   # keep 20% key points
        self.iterations = 5000
        self.error_threshold = 2

    def get_label(self):
        R_most_accurate = None
        T_most_accurate = None
        minimum_error = float('inf')
        cnt = 0
        for iter in tqdm(range(self.iterations), desc="Generating label ..."):
            # calculate the rough T
            point_1 = random.randint(0, self.num_of_points)
            point_2 = random.randint(0, self.num_of_points)
            point_3 = random.randint(0, self.num_of_points)
            point_4 = random.randint(0, self.num_of_points)

            pair_2d = np.array( [self.input_array[point_1][0], self.input_array[point_2][0], self.input_array[point_3][0], self.input_array[point_4][0]] )
            pair_3d = np.array( [self.input_array[point_1][1], self.input_array[point_2][1], self.input_array[point_3][1], self.input_array[point_4][1]] )

            # solve EPnP
            _, R_rough, T_rough = cv.solvePnP(pair_3d, pair_2d, self.camera_matrix, None, None, None, flags= cv.SOLVEPNP_EPNP)
            
            # projected the 3d point with R, T and find the minimum error T
            R_rough, _ = cv.Rodrigues(R_rough)
            T_rough_matrix = np.hstack((R_rough, T_rough))

            # find the inlier of the 3d point
            num_of_inlier, inlier_2d, inlier_3d = self.find_inlier( T_rough_matrix )

            # find T_accurate with inlier
            if num_of_inlier >= 4:
                _, R_acc, T_acc = cv.solvePnP(pair_3d, pair_2d, self.camera_matrix, None, None, None, flags= cv.SOLVEPNP_EPNP)
                R_acc, _ = cv.Rodrigues(R_acc)
                T_acc_matrix = np.hstack((R_acc, T_acc))
                error = self.calculate_error_by_projection(T_acc_matrix)
            
                # find the minimum error T&R
                if error < minimum_error:
                    cnt+=1
                    minimum_error = error
                    R_most_accurate = R_acc
                    T_most_accurate = T_acc
            
        print("Sum of error : ", minimum_error )
        T_most_acc_matrix = np.hstack((R_most_accurate, T_most_accurate))
        print("T: ", T_most_acc_matrix )
        # set inlier && outlier label by T
        error_list = self.set_label(T_most_acc_matrix)

        return error_list

    def find_inlier(self, T):
        inlier_3d = []
        inlier_2d = []
        K = self.camera_matrix
        for point in self.input_array:
            point_3d_homogeneous = np.array([point[1][0], point[1][1], point[1][2], 1])
            uv = ((K @ T) @ point_3d_homogeneous.T)
            project_u = uv[0] / uv[2]
            project_v = uv[1] / uv[2]
            error = math.sqrt( (project_u - point[0][0]) ** 2 + (project_v - point[0][1]) ** 2 )
            if error < self.error_threshold:
                inlier_2d.append([point[0][0], point[0][1]])
                inlier_3d.append([point[1][0], point[1][1], point[1][2]])

        return len(inlier_2d), inlier_2d, inlier_3d

    def calculate_error_by_projection(self, T):
        K = self.camera_matrix
        point_3d_homogeneous = np.array([[point[1][0], point[1][1], point[1][2], 1] for point in self.input_array], dtype=float)
        uv = np.dot(np.dot(K, T), point_3d_homogeneous.T)
        project_u = uv[0] / uv[2]
        project_v = uv[1] / uv[2]
        error = np.sum((project_u - np.array([point[0][0] for point in self.input_array])) ** 2 + (project_v - np.array([point[0][1] for point in self.input_array])) ** 2)
        return error
    
    def set_label(self, T):
        K = self.camera_matrix
        error_list = []
        for point in self.input_array:
            point_3d_homogeneous = np.array([point[1][0], point[1][1], point[1][2], 1])
            uv = ((K @ T) @ point_3d_homogeneous.T)
            project_u = uv[0] / uv[2]
            project_v = uv[1] / uv[2]
            error_list.append( math.sqrt( (project_u - point[0][0]) ** 2 + (project_v - point[0][1]) ** 2 ) )
    
        return error_list