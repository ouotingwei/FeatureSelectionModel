import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from tqdm import tqdm
from transformers import BertModel
import torch
import torch.nn as nn

import featurebooster as FEATUREBOOSTER
import training_input as TRAINING_INPUT
import orb_operation as ORB
import generate_label as GENERATE_LABEL

#dataset folder path( change to your own path )
color_img_file = '/home/wei/deep_feature_selection/data/small_coffee/color'
depth_image_file = '/home/wei/deep_feature_selection/data/small_coffee/aligned_depth'
gt_file = '/home/wei/deep_feature_selection/data/small_coffee/groundtruth.txt'
camera_intrinsics = [4.2214370727539062e+02, 4.2700833129882812e+02, 4.2214370727539062e+02, 2.4522090148925781e+02] # from sensors.yaml ???

output_folder = '/home/wei/deep_feature_selection/training_data/small_coffee'

if __name__ == '__main__':
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = BertModel.from_pretrained('bert-base-uncased')

    if not os.path.exists(color_img_file) or not os.path.exists(depth_image_file):
        print("the file is not exist")
        exit()

    data_preprocess = TRAINING_INPUT.data_preprocessing()
    color_img_list, depth_img_list, gt_data_list, sequence_list =  data_preprocess.get_data_list(color_img_file, depth_image_file, gt_file)

    num_of_features = []

    training_cnt = 1

    for i in range(len(color_img_list)-1):
        print(" [-] img : ", training_cnt)
        training_cnt += 1

        training_input = [] 

        # get orb keypoints and descriptor from orb_feature_extraction.py ( return keypoint1 & descriptor1 )
        now_img = cv.imread( color_img_list[i] ) 
        next_img = cv.imread( color_img_list[i+1] )
        now_depth_img = cv.imread( depth_img_list[i], cv.IMREAD_UNCHANGED )
        
        # Feature booster
        now_kp, now_des = FEATUREBOOSTER.booster_process( now_img )
        next_kp, next_des = FEATUREBOOSTER.booster_process( next_img )

        # convert the keypoint into cv format
        now_kp = FEATUREBOOSTER.convert_to_cv_keypoints(now_kp)
        next_kp = FEATUREBOOSTER.convert_to_cv_keypoints(next_kp)

        #now_image_ = orb.orb_features( now_img )
        #next_image_ = orb.orb_features( next_img )
        #now_kp, now_des = now_image_.feature_extract() 
        #next_kp, next_des = next_image_.feature_extract() 

        # matching the feature between two  image frames
        matcher_ = ORB.feature_match( now_img, next_img, now_kp, next_kp, now_des, next_des ) 
        queryIdx, trainIdx = matcher_.frame_match()
        num_of_features.append( len(trainIdx) )

        # insert features into training_input
        input_ = TRAINING_INPUT.set_training_input(queryIdx, trainIdx, now_kp, next_kp, camera_intrinsics, now_depth_img, now_img)
        training_input = input_.insert_input()

        label_ = GENERATE_LABEL.generate_label(training_input, now_img.shape, camera_intrinsics)
        error_list = label_.get_label()

        # save training input and labels
        folder_name = output_folder + '/' + str(i)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        training_uv = [np.array(data_col[0]) for data_col in training_input]
        training_XYZ = [np.array(data_col[1]) for data_col in training_input]
        training_diversity2d = [np.array(data_col[2]) for data_col in training_input]
        training_diversity3d = [np.array(data_col[3]) for data_col in training_input]
        error_list = np.array(error_list)

        np.save(folder_name + '/' + 'input_uv.npy', training_uv)
        np.save(folder_name + '/' + 'input_XYZ.npy', training_XYZ)
        np.save(folder_name + '/' + 'input_diversity2d.npy', training_diversity2d)
        np.save(folder_name + '/' + 'input_diversity3d.npy', training_diversity3d)
        np.save(folder_name + '/' + 'error.npy', error_list)
            

    print("min : ", min(num_of_features))
    
    plt.plot( num_of_features, marker='o', linestyle='-' )
    #plt.xticks( num_of_features )
    plt.show()