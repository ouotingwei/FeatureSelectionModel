import os
import sys
import numpy as np
import torch
import yaml
import cv2 as cv
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import transforms3d.euler as euler
import transforms3d.quaternions as quat

import orb_feature_extraction as ofe
from featurebooster import FeatureBooster 
import featurebooster as fb
import feature_match as fm
import training_input as ti

sys.path.append('code/training/extractors/orbslam2_features/lib')
from orbslam2_features import ORBextractor

#dataset folder path( change to your own path )
color_img_file = '/media/wei/T7/資料集/OpenLORIS-Scene/201911-package/cafe1-1_2-package/cafe1-1/color'
depth_image_file = '/media/wei/T7/資料集/OpenLORIS-Scene/201911-package/cafe1-1_2-package/cafe1-1/depth'
gt_file = "/media/wei/T7/資料集/OpenLORIS-Scene/201911-package/cafe1-1_2-package/cafe1-1/groundtruth.txt"
camera_intrinsics = [4.2214370727539062e+02, 4.2700833129882812e+02, 4.2214370727539062e+02, 2.4522090148925781e+02] # from sensors.yaml

def find_gt_by_interpolation(sequence, gt_data):
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

def get_data_list(color_file_path, depth_file_path, gt_file_path):
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
        gt_data_list.append( find_gt_by_interpolation(sequence, gt_data) )

    return color_img_list, depth_img_list, gt_data_list, sequence_list

if __name__ == '__main__':
    if not os.path.exists(color_img_file) or not os.path.exists(depth_image_file):
        print("the file is not exist")
        exit()

    color_img_list, depth_img_list, gt_data_list, sequence_list = get_data_list(color_img_file, depth_image_file, gt_file)
    
    color_files = os.listdir(color_img_file)
    depth_files = os.listdir(depth_image_file)

    num_of_features = []

    for i in range(len(color_files)-1):
        training_input = [] 

        # get orb keypoints and descriptor from orb_feature_extraction.py ( return keypoint1 & descriptor1 )
        now_img = cv.imread( color_img_list[i] ) 
        next_img = cv.imread( color_img_list[i+1] )
        now_depth_img = cv.imread( depth_img_list[i], cv.IMREAD_UNCHANGED )
        
        # Feature booster
        now_kp, now_des = fb.booster_process( now_img )
        next_kp, next_des = fb.booster_process( next_img )

        # convert the keypoint into cv format
        now_kp = fb.convert_to_cv_keypoints(now_kp)
        next_kp = fb.convert_to_cv_keypoints(next_kp)

        #now_image_ = ofe.orb_features( now_img )
        #next_image_ = ofe.orb_features( next_img )

        #now_kp, now_des = now_image_.feature_extract() 
        #next_kp, next_des = next_image_.feature_extract() 

        # matching the feature between two  image frames
        matcher = fm.feature_match( now_img, next_img, now_kp, next_kp, now_des, next_des ) 
        match_2d_point = matcher.frame_match()
        num_of_features.append( len(match_2d_point) )

        # insert features into training_input
        training_input = ti.insert_input( match_2d_point, now_kp, camera_intrinsics, now_depth_img, now_img )

        # model
        # input : n feature array in one serquence
        # output : n scores of each feature array


        # Solve PnP 
        # input : 


        # loss function

    print("min : ", min(num_of_features))
    
    plt.plot( num_of_features, marker='o', linestyle='-' )
    #plt.xticks( num_of_features )
    plt.show()