import os
import sys
import numpy as np
import torch
import yaml
import cv2 as cv
import time
import matplotlib.pyplot as plt

import orb_feature_extraction as ofe
from featurebooster import FeatureBooster 
import featurebooster as fb
import feature_match as fm
import training_input as ti

sys.path.append('code/training/extractors/orbslam2_features/lib')
from orbslam2_features import ORBextractor

#dataset folder path( change to your own path )
color_img_file = 'data/small_coffee/color'
depth_image_file = 'data/small_coffee/aligned_depth'
intrinsics = [ 4.2214370727539062e+02, 4.2700833129882812e+02, 4.2214370727539062e+02, 2.4522090148925781e+02 ] # from sensors.yaml

if __name__ == '__main__':
    if not os.path.exists(color_img_file) or not os.path.exists(depth_image_file):
        print("the file is not exist")
        exit()
    
    color_files = os.listdir(color_img_file)
    depth_files = os.listdir(depth_image_file)

    num_of_features = []

    for i in range(len(color_files)-1):
        training_input = [] 

        now_sequence = color_files[i].rsplit('.', 1)[0]
        next_sequence = color_files[i+1].rsplit('.', 1)[0]
        depth_sequence = depth_files[i].rsplit('.', 1)[0]

        now_img_path = os.path.join(color_img_file, now_sequence + ".png")
        next_img_path = os.path.join(color_img_file, next_sequence + ".png")
        depth_img_path = os.path.join(depth_image_file, depth_sequence + ".png")

        # get orb keypoints and descriptor from orb_feature_extraction.py ( return keypoint1 & descriptor1 )
        now_img = cv.imread( now_img_path ) 
        next_img = cv.imread( next_img_path )
        now_depth_img = cv.imread( depth_img_path, cv.IMREAD_UNCHANGED )
        
        # Feature booster
        now_kp, now_des = fb.booster_process( now_img )
        next_kp, next_des = fb.booster_process( next_img )

        # convert the keypoint into cv format
        now_kp = fb.convert_to_cv_keypoints(now_kp)
        next_kp = fb.convert_to_cv_keypoints(next_kp)

        '''
        now_image_ = ofe.orb_features( now_img )
        next_image_ = ofe.orb_features( next_img )

        now_kp, now_des = now_image_.feature_extract() 
        next_kp, next_des = next_image_.feature_extract() 
        '''

        # matching the feature between two  image frames
        matcher = fm.feature_match( now_img, next_img, now_kp, next_kp, now_des, next_des ) 
        match_2d_point = matcher.frame_match()
        num_of_features.append( len(match_2d_point) )

        # insert features into training_input
        training_input = ti.insert_input( match_2d_point, now_kp, intrinsics, now_depth_img, now_img)

        # model

        # Solve PnP 

        # loss function

    print("min : ", min(num_of_features))
    
    plt.plot( num_of_features, marker='o', linestyle='-' )
    #plt.xticks( num_of_features )
    plt.show()