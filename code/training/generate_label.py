import numpy as np
import time
import cv2 as cv
from itertools import combinations
from tqdm import tqdm
from numba import jit

class generate_label:
    def __init__(self, input_array, img_size, camera_intrinsic):
        self.input_array = input_array
        self.camera_matrix = np.array([[camera_intrinsic[0], 0, camera_intrinsic[2]], [0, camera_intrinsic[1], camera_intrinsic[3]], [0, 0, 1]], dtype=float)
        self.h = img_size[0]
        self.w = img_size[1]
        self.keep_ratio = 0.2   # keep 20% key points

    @jit
    def get_label(self):
        input_length = len(self.input_array)
        combination_sequence = np.arange(0, input_length )
        # combination of N Choose 3, N is the number of key points
        combinations_list = list(combinations(combination_sequence, 3))
        # numpy -> cv.UMat
        camera_matrix = cv.UMat(self.camera_matrix)

        R_most_accurate = None
        T_most_accurate = None
        minimum_error = float('inf')

        for combination in tqdm(combinations_list, desc="Generating the labels..."):
            point_1 = combination[0]
            point_2 = combination[1]
            point_3 = combination[2]

            pair_2d = [self.input_array[point_1][0], self.input_array[point_2][0], self.input_array[point_3][0]]
            pair_3d = [self.input_array[point_1][1], self.input_array[point_2][1], self.input_array[point_3][1]]

            # object_point -> NumPy -> UMat
            image_points_np = np.ascontiguousarray(np.array(pair_2d, dtype=np.float32))
            image_points_um = cv.UMat(image_points_np)
            object_points_np = np.ascontiguousarray(np.array(pair_3d, dtype=np.float32))
            object_points_um = cv.UMat(object_points_np)

            # solve P3P -> T ( up to four solutions but I don't know why )
            retval, R, T = cv.solveP3P(object_points_um, image_points_um, camera_matrix, None, flags= cv.SOLVEPNP_P3P)

            # projected the 3d point with R, T and find the minimum error T
            for i in range(retval):
                R_matrix, _ = cv.Rodrigues(R[i])
                T_matrix = np.hstack((R_matrix, T[i]))
                error = self.calculate_error_by_projection(T_matrix)
                
                # find the minimum error T&R
                if error < minimum_error:
                    minimum_error = error
                    R_most_accurate = R[i]
                    T_most_accurate = T[i]
            
        print(minimum_error)
        # set inlier && outlier label by T


        return None
    @jit
    def calculate_error_by_projection(self, T):
        K = self.camera_matrix
        point_3d_homogeneous = np.array([[point[1][0], point[1][1], point[1][2], 1] for point in self.input_array], dtype=float)
        uv = np.dot(np.dot(K, T), point_3d_homogeneous.T)
        project_u = uv[0] / uv[2]
        project_v = uv[1] / uv[2]
        error = np.sum((project_u - np.array([point[0][0] for point in self.input_array])) ** 2 + (project_v - np.array([point[0][1] for point in self.input_array])) ** 2)
        return error







    def set_label(self):
        label = []  

        return label