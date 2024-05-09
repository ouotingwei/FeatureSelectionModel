import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time
import sys

sys.path.append('/home/wei/deep_feature_selection/code/training/extractors/orbslam2_features/lib')
from orbslam2_features import ORBextractor

class orb_features:
    def __init__(self, img):
        self.img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        self.keypoint = None
        self.descriptor = None
        self.feature_extractor = ORBextractor(1000, 1.2, 8)
    
    def feature_warning( self ):
        num_keypoints = len(self.descriptor)
        if num_keypoints < 10 :
            print("\033[93m[Feature Extract Warning] Not Enough Number of keypoints : {}\033[0m".format(num_keypoints))
 
        else:
            print("\033[92m[Feature Extract] Number of keypoints: {}\033[0m".format(num_keypoints))
    
    def feature_extract(self):
        start_time = time.time()

        # Initiate ORB detector
        #orb = cv.ORB_create(nfeatures=500)

        # find the keypoints with ORB
        #self.keypoint = orb.detect(self.img, None)

        # compute the descriptors with ORB
        self.keypoint, self.descriptor = self.feature_extractor.detectAndCompute(self.img)

        end_time = time.time()

        orb_feature_extract_time = end_time - start_time
        print("[Feature Extraction] time: {:.6f} seconds".format(orb_feature_extract_time))

        self.feature_warning()

        # draw only keypoints location,not size and orientation
        # show = cv.drawKeypoints(self.img, self.keypoint, None, color=(0,255,0), flags=0)
        # plt.imshow(show), plt.show()

        return self.keypoint, self.descriptor
    
'''
match_feature need to be optimized after first training
'''
class feature_match:
    def __init__(self, img1, img2, keypoint1, keypoint2, descriptor1, descriptor2):
        # input images
        self.img1 = cv.cvtColor(img1, cv.COLOR_RGB2BGR)
        self.img2 = cv.cvtColor(img2, cv.COLOR_RGB2BGR)

        # input keypoints
        self.keypoint1 = keypoint1
        self.keypoint2 = keypoint2

        # input descriptors
        self.descriptor1 = descriptor1
        self.descriptor2 = descriptor2

    def frame_match(self):
        # Create BFMatcher object with cross check
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        
        # Match descriptors
        matches = bf.match(self.descriptor1, self.descriptor2)

        min_dist = 10000
        max_dist = 0

        for match in matches:
            dist = match.distance
            if dist < min_dist and dist != 0:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist

        print("Found minimum distance", min_dist, max_dist)
        
        # Filter matches based on the Hamming distance
        good_matches = []
        matched_2d_points = []
        for match in matches:
            if match.distance <= 10 * min_dist:
            #if match.distance <= 50:
                good_matches.append(match)
                matched_2d_points.append(match.trainIdx)
                #print(match.trainIdx, match.queryIdx) #trainIdx=descriptor2
        
        print("There are ", len(good_matches), 'Points with good match')

        # Draw only good matches
        img3 = cv.drawMatches(np.uint8(self.img1), self.keypoint1, np.uint8(self.img2), self.keypoint2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Display the result
        plt.imshow(img3)
        plt.show()

        return matched_2d_points