import os
import sys
import numpy as np
import torch
import yaml
import cv2 as cv
import time

import orb_feature_extraction as ofe
from featurebooster import FeatureBooster 
import featurebooster as fb
import feature_match as fm

sys.path.append('/home/wei/deep_feature_selection/code/training/extractors/orbslam2_features/lib')
from orbslam2_features import ORBextractor

#dataset folder path
color_img_file = '/home/wei/deep_feature_selection/data/small_coffee/color'
depth_image_file = '/home/wei/deep_feature_selection/data/small_coffee/aligned_depth'

if __name__ == '__main__':
    if not os.path.exists(color_img_file) or not os.path.exists(depth_image_file):
        print("the file is not exist")
        exit()
    
    color_files = os.listdir(color_img_file)

    for i in range(len(color_files)-1):
        now_sequence = color_files[i].rsplit('.', 1)[0]
        next_sequence = color_files[i+1].rsplit('.', 1)[0]

        # get orb keypoints and descriptor from orb_feature_extraction.py ( return keypoint1 & descriptor1 )
        now_img = cv.imread( os.path.join(color_img_file, now_sequence + ".png") ) 
        next_img = cv.imread( os.path.join(color_img_file, next_sequence + ".png") )

        # 
        now_kp, now_des = fb.booster_process( now_img )
        next_kp, next_des = fb.booster_process( next_img )

        # convert the keypoint into cv format
        now_kp = fb.convert_to_cv_keypoints(now_kp)
        next_kp = fb.convert_to_cv_keypoints(next_kp)

        # matching the feature between two  image frames
        matcher = fm.feature_match( now_img, next_img, now_kp, next_kp, now_des, next_des ) 
        matcher.frame_match()

        # get depth of the feature points

        # get sparity of the feature points

        # get semantic info of the feature points


    

    