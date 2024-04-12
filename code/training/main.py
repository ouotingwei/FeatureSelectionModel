import cv2 as cv
import orb_feature_extraction as ofe

if __name__ == '__main__':
    # get orb features from img1
    img1 = ofe.orb_features( cv.imread("/home/wei/deep_feature_selection/data/test_img/1_c.png") )
    descriptor1 = img1.feature_extract() 

    # get orb features from img2
    img2 = ofe.orb_features( cv.imread("/home/wei/deep_feature_selection/data/test_img/IMG_3876.jpeg") )
    descriptor2 = img2.feature_extract() 