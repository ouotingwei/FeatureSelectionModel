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
        orb = cv.ORB_create(nfeatures=500)

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

        print("Keypoint position (x, y):", self.keypoint[0].pt)
        print("Keypoint size:", self.keypoint[0].size)
        print("Keypoint angle:", self.keypoint[0].angle)
        print("Keypoint response:", self.keypoint[0].response)
        print("Keypoint octave:", self.keypoint[0].octave)
        print("Keypoint class ID:", self.keypoint[0].class_id)

        print("descriptor", self.descriptor[0])


        return self.keypoint, self.descriptor