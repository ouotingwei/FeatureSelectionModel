import cv2 as cv
import orb_feature_extraction as ofe
import feature_match as fm

if __name__ == '__main__':
    # get orb keypoints and descriptor from orb_feature_extraction.py ( return keypoint1 & descriptor1 )
    img1 = cv.imread( "/home/wei/deep_feature_selection/data/test_img/test1/1563004641.130956.png" )
    image1_ = ofe.orb_features( img1 )
    keypoint1, descriptor1 = image1_.feature_extract() 

    # get orb keypoints and descriptor from orb_feature_extraction.py ( return keypoint2 & descriptor2 )
    img2 = cv.imread( "/home/wei/deep_feature_selection/data/test_img/test1/1563004641.164202.png" )
    image2_ = ofe.orb_features( img2 )
    keypoint2, descriptor2 = image2_.feature_extract() 

    # matching the feature between two  image frames
    matcher = fm.feature_match( img1, img2, keypoint1, keypoint2, descriptor1, descriptor2) 
    matcher.frame_match()

    # feature booster


    # get vector

    

    