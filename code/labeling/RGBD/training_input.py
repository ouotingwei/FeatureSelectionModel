import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
import os
import sys

import semantic as semantic

class set_training_input:
    def __init__(self, queryIdx, trainIdx, now_kp, next_kp, intrinsic, now_depth_img, now_rgb_img):
        self.nowIdx = queryIdx
        self.nextIdx = trainIdx
        self.now_kp = now_kp
        self.next_kp = next_kp
        self.intrinsic = intrinsic
        self.now_depth_img = now_depth_img
        self.now_rgb_img = now_rgb_img

        self.depth_scale = 1000
        self.patch_w = 84.8
        self.patch_h = 48

        self.h = now_depth_img.shape[0]
        self.w = now_depth_img.shape[1]
    
    def insert_input(self):
        training_input = []
        new_uv = []
        
        for i in range(len(self.nextIdx)):
            now_uv = self.new_uv_input(i)
            uv = self.uv_input(i)
            XYZ = self.XYZ_input(i)
            MD = self.get_mean_depth_around_kp_input(i)
            NB = self.get_numbers_of_nearby_kp(i)
            RES = self.new_response(i)
            SIZE = self.new_size(i)
            
            if XYZ != [0.0, 0.0, 0.0] and MD != [sys.maxsize]:
                training_input.append([uv, XYZ, MD, NB, RES, SIZE])
                new_uv.append(now_uv)
            '''
            if XYZ != [0.0, 0.0, 0.0]:
                training_input.append([uv, XYZ, NB, RES, SIZE])
                new_uv.append(now_uv)
            '''
        
        '''
        ok_training_input = []
        for i in range( len(self.nextIdx) ):
            if training_input[i][1] != [0.0, 0.0, 0.0] and training_input[i][1] != [0.0, -0.0, 0.0] and training_input[i][2] != 1000.0:
                ok_training_input.append( training_input[i] )
        

        print("number of key points : ", len(ok_training_input) )
        '''

        return training_input, new_uv
    
    def new_response ( self, idx ):
        response = self.now_kp[ self.nowIdx[idx] ].response
        return[response]
    
    def new_size( self, idx):
        orb_size = self.now_kp[ self.nowIdx[idx] ].size
        return[orb_size]
    
    def new_uv_input( self, idx ):
        '''
        u & v = trainIdx(next)
        '''
        u = self.now_kp[ self.nowIdx[idx] ].pt[0] 
        v = self.now_kp[ self.nowIdx[idx] ].pt[1]

        #return [u/self.w, v/self.h]
        return [u, v]

    def uv_input( self, idx ):
        '''
        u & v = trainIdx(next)
        '''
        u = self.next_kp[ self.nextIdx[idx] ].pt[0] 
        v = self.next_kp[ self.nextIdx[idx] ].pt[1]

        #return [u/self.w, v/self.h]
        return [u, v]

    def XYZ_input( self, idx ):
        '''
        u, v = queryIdx(now)
        X, Y, Z = queryIdx(previous)
        '''
        u = self.now_kp[ self.nowIdx[idx] ].pt[0] 
        v = self.now_kp[ self.nowIdx[idx] ].pt[1]
        #print(u, v)

        Z =  self.now_depth_img[int(v), int(u)]  / self.depth_scale
        X = ( u - self.intrinsic[2] ) * Z / self.intrinsic[0]
        Y = ( v - self.intrinsic[3] ) * Z / self.intrinsic[1]

        return [X, Y, Z]

    # must be implemented
    def get_mean_depth_around_kp_input(self, idx):
        '''
        u & v = now
        [ u, v ] -> Z =  depth_img[int(v), int(u)]  / depth_scale
        [1][2][3]
        [4][5][6]
        [7][8][9]
        '''
        u = self.now_kp[self.nowIdx[idx]].pt[0]
        v = self.now_kp[self.nowIdx[idx]].pt[1]
        Z = self.now_depth_img[int(v), int(u)] / self.depth_scale

        # 初始化深度累計值和計數器
        total_depth = 0
        count = 0
        
        # 遍歷3x3鄰域
        for i in range(-1, 2):
            for j in range(-1, 2):
                u_neigh = int(u) + i
                v_neigh = int(v) + j
                if 0 <= u_neigh < self.now_depth_img.shape[1] and 0 <= v_neigh < self.now_depth_img.shape[0]:
                    depth_value = self.now_depth_img[v_neigh, u_neigh]
                    if depth_value != 0.0:
                        total_depth += depth_value
                        count += 1

        if count == 0:
            return [sys.maxsize]

        mean_depth = total_depth / count

        center_depth = self.now_depth_img[int(v), int(u)]

        if center_depth == 0.0:
            return float(sys.maxsize)

        depth_diff = abs(center_depth - mean_depth) / self.depth_scale

        return [float(depth_diff)]

    def get_numbers_of_nearby_kp(self, idx):
        '''
        u & v = next
        '''
        u = self.now_kp[ self.nowIdx[idx] ].pt[0] 
        v = self.now_kp[ self.nowIdx[idx] ].pt[1]

        # kp range
        u_min = u - self.patch_w/2
        u_max = u + self.patch_w/2
        v_min = v - self.patch_h/2
        v_max = v + self.patch_h/2

        # num of kp in one patch
        cnt = sum(
            1 for i in self.nowIdx
            if u_min < self.now_kp[i].pt[0] < u_max and v_min < self.now_kp[i].pt[1] < v_max
        )

        cnt -= 1
        return [max(0, cnt)]


class data_preprocessing:
    def __init__(self):
        pass

    def get_intrinsic_from_ymal(self):
        pass

    def get_data_list(self, color_file_path, depth_file_path):
        '''
        input : color_file, depth_file, gt_file
        output : color_path_list, depth_path_list, gt_data_list
        '''

        color_files = sorted(os.listdir(color_file_path), key=lambda x: float(os.path.splitext(x)[0]))
        depth_files = sorted(os.listdir(depth_file_path), key=lambda x: float(os.path.splitext(x)[0]))
        sequence = color_files

        color_img_list = []
        depth_img_list = []
        sequence_list = []

        for i in tqdm(range(len(color_files)), "loading data ..."):
            now_img = color_files[i].rsplit('.', 1)[0]
            now_depth = depth_files[i].rsplit('.', 1)[0]

            now_img_path = os.path.join(color_file_path, now_img + ".png")
            now_depth_path = os.path.join(depth_file_path, now_depth + ".png")

            color_img_list.append(now_img_path)
            depth_img_list.append(now_depth_path)

            sequence = float(now_img)
            sequence_list.append(sequence)

        return color_img_list, depth_img_list, sequence_list 