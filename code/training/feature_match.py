import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time

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
        for match in matches:
            if match.distance <= 2 * min_dist:
            #if match.distance <= 50:
                good_matches.append(match)
        
        print("There are ", len(good_matches), 'Points with good match')
        
        # Draw only good matches
        img3 = cv.drawMatches(np.uint8(self.img1), self.keypoint1, np.uint8(self.img2), self.keypoint2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Display the result
        plt.imshow(img3)
        plt.show()

