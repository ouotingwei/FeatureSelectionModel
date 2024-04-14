import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time

class orb_features:
    def __init__(self, img):
        self.img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.keypoint = None
        self.descriptor = None
    
    def feature_warning( self ):
        num_keypoints = len(self.descriptor)
        if num_keypoints < 10 :
            print("\033[93m[Feature Extract Warning] Not Enough Number of keypoints : {}\033[0m".format(num_keypoints))
 
        else:
            print("\033[92m[Feature Extract] Number of keypoints: {}\033[0m".format(num_keypoints))
    
    def feature_extract(self):
        start_time = time.time()

        # Initiate ORB detector
        orb = cv.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)

        # find the keypoints with ORB
        self.keypoint = orb.detect(self.img, None)

        # compute the descriptors with ORB
        self.keypoint, self.descriptor = orb.compute(self.img, self.keypoint)

        end_time = time.time()

        orb_feature_extract_time = end_time - start_time
        print("[Feature Extraction] time: {:.6f} seconds".format(orb_feature_extract_time))

        self.feature_warning()

        # draw only keypoints location,not size and orientation
        show = cv.drawKeypoints(self.img, self.keypoint, None, color=(0,255,0), flags=0)
        plt.imshow(show), plt.show()

        return self.keypoint, self.descriptor