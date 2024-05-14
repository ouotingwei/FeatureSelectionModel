import numpy as np
import time
import cv2 as cv
from itertools import combinations

'''
思路：
1. 隨機取四個點計算EPnP，做N次
2. 將所有3D特徵點根據第一步計算出的T投影成2D座標後計算誤差
3. 取誤差最小的那組T，並保留50%內點
4. 賦予其正負類別的Label
'''

class generate_label:
    def __init__(self, input_array, img_size, camera_intrinsic):
        self.input_array = input_array
        self.minimum_error_T = None
        self.keep_ratio = 0.2
        self.error = float('inf')

    def get_label(self):
        input_length = len(self.input_array)

        combination_sequence = [i for i in range(1, input_length + 1)]

        combinations_list = list(combinations(combination_sequence, 3))

        print(len(combinations_list))

        for i in range(1):
            point_1 = combinations_list[i][0]
            point_2 = combinations_list[i][1]
            point_3 = combinations_list[i][2]
            pair_2d = [self.input_array[point_1][0], self.input_array[point_2][0], self.input_array[point_3][0]]
            pair_3d = [self.input_array[point_1][1], self.input_array[point_2][1], self.input_array[point_3][1]]

            # solve P3P -> T

            # projected the 3d point with T

            # Calcultate the error 

            # if error < self.error -> self.error = error && save T
            
        
        # set inlier && outlier label by T

        return None


    def pnp_solver(self, pair_2d, pair_3d):
        _,R,T=cv.solvePnP(objp,corners,mtx,dist)

    def set_label(self, error_list):
        label = []  
        for i in range( len(error_list) ):
            pass

        return label