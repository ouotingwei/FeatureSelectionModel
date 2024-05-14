import os
import cv2 as cv
import matplotlib.pyplot as plt
from typing import List
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
camera_intrinsics = [4.2214370727539062e+02, 4.2700833129882812e+02, 4.2214370727539062e+02, 2.4522090148925781e+02] # from sensors.yaml

def fing_inliers():
    pass

if __name__ == '__main__':
    print(cv.__version__)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = BertModel.from_pretrained('bert-base-uncased')

    if not os.path.exists(color_img_file) or not os.path.exists(depth_image_file):
        print("the file is not exist")
        exit()

    data_preprocess = TRAINING_INPUT.data_preprocessing()
    color_img_list, depth_img_list, gt_data_list, sequence_list =  data_preprocess.get_data_list(color_img_file, depth_image_file, gt_file)

    num_of_features = []

    for i in range(len(color_img_list)-1):
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

        label_ = GENERATE_LABEL.generate_label(training_input, None, None)
        label = label_.get_label()

        # model
        # input : n feature array in one serquence
        # output : n scores of each feature array

    print("min : ", min(num_of_features))
    
    plt.plot( num_of_features, marker='o', linestyle='-' )
    #plt.xticks( num_of_features )
    plt.show()