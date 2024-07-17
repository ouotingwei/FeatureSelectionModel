'''
/home/ee605-wei/small_test
'''
import orb_operation as ORB
import training_input as TRAINING_INPUT
import cv2 as cv
import numpy as np
import time

color_file_path = '/media/ee605-wei/ROG_512_SSD/lab605_10ms/color'
data_preprocess = TRAINING_INPUT.data_preprocessing()
color_img_list = data_preprocess.get_data_list(color_file_path, '.png')

now_img = cv.imread(color_img_list[0]) 
next_img = cv.imread(color_img_list[1])

# ORB Feature Extraction
now_image_ = ORB.orb_features(now_img) 
now_kp, now_des = now_image_.feature_extract() 
next_image_ = ORB.orb_features(next_img)
next_kp, next_des = next_image_.feature_extract() 

# matching the feature between two  image frames
matcher_ = ORB.feature_match(now_img, next_img, now_kp, next_kp, now_des, next_des) 
queryIdx, trainIdx = matcher_.frame_match()  # now->query, next->train

base_kp = [now_kp[i] for i in queryIdx]
base_des = cv.UMat(np.array([now_des[i] for i in queryIdx]))

match_list = np.ones(len(base_kp))
for i in range(2, len(color_img_list)):
    ne_img = cv.imread(color_img_list[i])

    # ORB Feature Extraction
    ne_image_ = ORB.orb_features(ne_img)
    ne_kp, ne_des = ne_image_.feature_extract() 

    # matching the feature between two  image frames
    matcher_ = ORB.feature_match(now_img, ne_img, base_kp, ne_kp, base_des, ne_des) 
    quIdx, _ = matcher_.frame_match()  # now->query, next->train

    for k in range(len(quIdx)):
        match_list[quIdx[k]] += 1
    
    cnt = 0

    for j in range(len(match_list)):
        if match_list[j] == i:

            cnt += 1
    print(match_list)
    rate = cnt/len(match_list)*100
    print(rate)
    time.sleep(1)
