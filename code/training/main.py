import os
import sys
import numpy as np
import torch
import yaml
from pathlib import Path
import cv2 as cv
import argparse

from featurebooster import FeatureBooster
import feature_match as fm
sys.path.append('/home/wei/deep_feature_selection/code/training/extractors/orbslam2_features/lib')
from orbslam2_features import ORBextractor

def normalize_keypoints(keypoints, image_shape):
    x0 = image_shape[1] / 2
    y0 = image_shape[0] / 2
    scale = max(image_shape) * 0.7
    kps = np.array(keypoints)
    kps[:, 0] = (keypoints[:, 0] - x0) / scale
    kps[:, 1] = (keypoints[:, 1] - y0) / scale
    return kps 

def booster_seprocess(image):
    # orb extractor
    feature_extractor = ORBextractor(500, 1.2, 8)

    # set FeatureBooster
    config_file = '/home/wei/deep_feature_selection/code/training/config.yaml'
    with open(str(config_file), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    feature_booster = FeatureBooster(config['ORB+Boost-B'])
    if use_cuda:
        feature_booster.cuda()
    feature_booster.eval()

    feature_booster.load_state_dict(torch.load('/home/wei/deep_feature_selection/code/training/ORB+Boost-B.pth'))

    kps_tuples, descriptors = feature_extractor.detectAndCompute(image)
    # convert keypoints 
    keypoints = [cv.KeyPoint(*kp) for kp in kps_tuples]
    keypoints = np.array(
        [[kp.pt[0], kp.pt[1], kp.size / 31, np.deg2rad(kp.angle)] for kp in keypoints], 
        dtype=np.float32
    )

    # boosted the descriptor using trained model
    kps = normalize_keypoints(keypoints, image.shape)
    kps = torch.from_numpy(kps.astype(np.float32))
    descriptors = np.unpackbits(descriptors, axis=1, bitorder='little')
    descriptors = descriptors * 2.0 - 1.0
    descriptors = torch.from_numpy(descriptors.astype(np.float32))

    if use_cuda:
        kps = kps.cuda()
        descriptors = descriptors.cuda()

    out = feature_booster(descriptors, kps)
    out = (out >= 0).cpu().detach().numpy()
    descriptors = np.packbits(out, axis=1, bitorder='little')

    return keypoints, descriptors

def convert_to_cv_keypoints(keypoints):
    cv_keypoints = []
    for kp in keypoints:
        x, y = kp[0], kp[1]
        size = kp[2]
        angle = kp[3]
        cv_kp = cv.KeyPoint(x, y, size, angle)
        cv_keypoints.append(cv_kp)
    return cv_keypoints

if __name__ == '__main__':
    # set CUDA
    use_cuda = torch.cuda.is_available()

    # set torch grad ( speed up ! )
    torch.set_grad_enabled(False)

    # get orb keypoints and descriptor from orb_feature_extraction.py ( return keypoint1 & descriptor1 )
    img1 = cv.imread( "/home/wei/deep_feature_selection/data/test_img/test1/1563004641.130956.png" )
    #image1_ = ofe.orb_features( img1 )
    #keypoint1, descriptor1 = image1_.feature_extract() 
    keypoint1, descriptor1 = booster_seprocess(img1)
    keypoint1 = convert_to_cv_keypoints(keypoint1)

    # get orb keypoints and descriptor from orb_feature_extraction.py ( return keypoint2 & descriptor2 )
    img2 = cv.imread( "/home/wei/deep_feature_selection/data/test_img/test1/1563004641.164202.png" )
    #image2_ = ofe.orb_features( img2 )
    #keypoint2, descriptor2 = image2_.feature_extract() 
    keypoint2, descriptor2 = booster_seprocess(img2)
    keypoint2 = convert_to_cv_keypoints(keypoint2)

    # matching the feature between two  image frames
    matcher = fm.feature_match( img1, img2, keypoint1, keypoint2, descriptor1, descriptor2) 
    matcher.frame_match()


    # get vector

    

    