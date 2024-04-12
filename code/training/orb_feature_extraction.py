import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time

class orb_features:
    def __init__(self, img):
        self.img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.descriptor = None
    
    def feature_warning( self ):
        num_keypoints = len(self.descriptor)
        if num_keypoints < 10 :
            print("\033[93m[Feature Extract Warning] Not Enough Number of keypoints : {}\033[0m".format(num_keypoints))
 
        else:
            print("\033[92m[Feature Extract] Number of keypoints: {}\033[0m".format(num_keypoints))
    
    def feature_extract(self):
        start_time = time.time()

        num_cores = cv.getNumberOfCPUs()

        # Initiate ORB detector
        orb = cv.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)

        # find the keypoints with ORB
        self.descriptor = orb.detect(self.img, None)

        # compute the descriptors with ORB
        self.descriptor, des = orb.compute(self.img, self.descriptor)

        end_time = time.time()

        orb_feature_extract_time = end_time - start_time
        print("[Feature Extraction] time: {:.6f} seconds".format(orb_feature_extract_time))

        self.feature_warning()

        # draw only keypoints location,not size and orientation
        show = cv.drawKeypoints(self.img, self.descriptor, None, color=(0,255,0), flags=0)
        plt.imshow(show), plt.show()

        return self.descriptor
    
    def frame_match(sekf):
        return 0