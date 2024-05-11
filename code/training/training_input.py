import numpy as np
from scipy.interpolate import interp1d
import os

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
        self.patch_w = 64
        self.patch_h = 48

        self.h = now_depth_img.shape[0]
        self.w = now_depth_img.shape[1]
    
    def insert_input(self):
        training_input = []
        
        for i in range( len(self.nextIdx) ):
            uv = self.uv_input(i)
            XYZ = self.XYZ_input(i)
            MD = self.get_mean_depth_around_kp_input(i)
            NB = self.get_numbers_of_nearby_kp(i)

            print( [ uv, XYZ, MD, NB ] )

            training_input.append( [uv,XYZ,MD,NB] )

        return training_input

    def uv_input( self, idx ):
        '''
        u & v = trainIdx(next)
        '''
        u = self.next_kp[ self.nextIdx[idx] ].pt[0] 
        v = self.next_kp[ self.nextIdx[idx] ].pt[1]

        return [u/self.w, v/self.h]

    def XYZ_input( self, idx ):
        '''
        u, v = queryIdx(now)
        X, Y, Z = queryIdx(previous)
        '''
        u = self.now_kp[ self.nowIdx[idx] ].pt[0] 
        v = self.now_kp[ self.nowIdx[idx] ].pt[1]

        Z =  self.now_depth_img[int(v), int(u)]  / self.depth_scale
        X = ( u - self.intrinsic[2] ) * Z / self.intrinsic[0]
        Y = ( v - self.intrinsic[3] ) * Z / self.intrinsic[1]

        return [X, Y, Z]

    def get_mean_depth_around_kp_input(self, idx):
        '''
        u & v = now
        [ u, v ] -> Z =  depth_img[int(v), int(u)]  / depth_scale
        [1][2][3]
        [4][5][6]
        [7][8][9]
        '''
        u = self.now_kp[ self.nowIdx[idx] ].pt[0] 
        v = self.now_kp[ self.nowIdx[idx] ].pt[1]
        Z = self.now_depth_img[int(v), int(u)]  / self.depth_scale

        rows, cols = np.indices((3, 3)) - 1

        row_indices = np.int32(u) + rows
        col_indices = np.int32(v) + cols

        # true -> indices is in the boundary(img)
        valid_indices = (row_indices >= 0) & (row_indices < self.h) & (col_indices >= 0) & (col_indices < self.w)

        # calculate the mean_depth
        valid_depths = self.now_depth_img[row_indices[valid_indices], col_indices[valid_indices]]
        mean_depth = np.mean(valid_depths) / self.depth_scale if len(valid_depths) > 0 else None

        if mean_depth is not None:
            diff = abs(mean_depth - Z)
        else:
            diff = None

        return [diff]

    def get_numbers_of_nearby_kp(self, idx):
        '''
        u & v = next
        '''
        u = self.next_kp[ self.nextIdx[idx] ].pt[0] 
        v = self.next_kp[ self.nextIdx[idx] ].pt[1]

        # kp range
        u_min = u - self.patch_w/2
        u_max = u + self.patch_w/2
        v_min = v - self.patch_h/2
        v_max = v + self.patch_h/2

        # num of kp in one patch
        cnt = sum(
            1 for i in self.nextIdx
            if u_min < self.next_kp[i].pt[0] < u_max and v_min < self.next_kp[i].pt[1] < v_max
        )

        cnt -= 1
        return [max(0, cnt)]


class data_preprocessing:
    def __init__(self):
        pass

    def get_intrinsic_from_ymal(self):
        pass

    def get_data_list(self, color_file_path, depth_file_path, gt_file_path):
        '''
        input : color_file, depth_file, gt_file
        output : color_path_list, depth_path_list, gt_data_list
        '''
        print("Loading data ...")

        color_files = sorted(os.listdir(color_file_path), key=lambda x: float(os.path.splitext(x)[0]))
        depth_files = sorted(os.listdir(depth_file_path), key=lambda x: float(os.path.splitext(x)[0]))
        sequence = color_files

        gt_data = []
        with open(gt_file_path, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    continue
                parts = line.strip().split(' ')
                time = float(parts[0])
                gt_entry = (time, *map(float, parts[1:]))
                gt_data.append(gt_entry)

        color_img_list = []
        depth_img_list = []
        gt_data_list = []
        sequence_list = []

        for i in range(len(color_files)):
            now_img = color_files[i].rsplit('.', 1)[0]
            now_depth = depth_files[i].rsplit('.', 1)[0]

            now_img_path = os.path.join(color_file_path, now_img + ".png")
            now_depth_path = os.path.join(depth_file_path, now_depth + ".png")

            color_img_list.append(now_img_path)
            depth_img_list.append(now_depth_path)

            sequence = float(now_img)
            sequence_list.append(sequence)
            gt_data_list.append( self.find_gt_by_interpolation(sequence, gt_data) )

        return color_img_list, depth_img_list, gt_data_list, sequence_list 
    
    def find_gt_by_interpolation(self, sequence, gt_data):
        closet_index = None
        for i in range(len(gt_data)):
            if gt_data[i][0] > sequence:
                closet_index = i
                break

        big_index = closet_index
        small_index = closet_index - 1
        t_ = [gt_data[small_index][0], gt_data[big_index][0]]
        x_ = [gt_data[small_index][1], gt_data[big_index][1]]
        y_ = [gt_data[small_index][2], gt_data[big_index][2]]
        quaz_ = [gt_data[small_index][6], gt_data[big_index][6]]
        quaw_ = [gt_data[small_index][7], gt_data[big_index][7]]

        interp_x = interp1d(t_, x_, kind='linear')
        now_x = interp_x(sequence)

        interp_y = interp1d(t_, y_, kind='linear')
        now_y = interp_y(sequence)

        interp_quaz = interp1d(t_, quaz_, kind='linear')
        now_quaz = interp_quaz(sequence)

        interp_quaw = interp1d(t_, quaw_, kind='linear')
        now_quaw = interp_quaw(sequence)

        return [now_x, now_y, 0.0, 0.0, 0.0, now_quaz, now_quaw]