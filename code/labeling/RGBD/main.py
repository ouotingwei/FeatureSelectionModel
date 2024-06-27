import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import training_input as TRAINING_INPUT
import orb_operation as ORB
import generate_label as GENERATE_LABEL

# dataset folder path( change to your own path )
color_file_path = '/media/ee605-wei/T7/資料集/training_800/1-2/color'
depth_file_path = '/media/ee605-wei/T7/資料集/training_800/1-2/depth'
camera_intrinsics = [611.4509887695312, 611.4857177734375, 433.2039794921875, 249.4730224609375] # fx, fy, cx, cy
output_folder = '/home/ee605-wei/FeatureSelectionModel/training_data/loris_800'

if __name__ == '__main__':
    if not os.path.exists(color_file_path) or not os.path.exists(depth_file_path):
        print("file is not exist !")
        exit()

    # image path -> list
    data_preprocess = TRAINING_INPUT.data_preprocessing()
    color_img_list, depth_img_list, sequence_list =  data_preprocess.get_data_list(color_file_path, depth_file_path)

    num_of_features = []
    error = []
    training_cnt = 0

    for i in range(len(color_img_list)-1):
        print(" [-] img : ", training_cnt)
        training_cnt += 1
        training_input = [] 

        # get orb keypoints and descriptor from orb_feature_extraction.py ( return keypoint1 & descriptor1 )
        now_img = cv.imread( color_img_list[i] ) 
        next_img = cv.imread( color_img_list[i+1] )
        now_depth_img = cv.imread( depth_img_list[i], cv.IMREAD_UNCHANGED )

        print("Current image size (now_img):", now_img.shape)
        print("Next image size (next_img):", next_img.shape)
        print("Current depth image size (now_depth_img):", now_depth_img.shape)
        
        # ORB Feature Extraction
        now_image_ = ORB.orb_features( now_img )
        next_image_ = ORB.orb_features( next_img )
        now_kp, now_des = now_image_.feature_extract() 
        next_kp, next_des = next_image_.feature_extract() 

        # matching the feature between two  image frames
        matcher_ = ORB.feature_match( now_img, next_img, now_kp, next_kp, now_des, next_des ) 
        queryIdx, trainIdx = matcher_.frame_match()
        num_of_features.append( len(trainIdx) )

        # insert features into training_input
        input_ = TRAINING_INPUT.set_training_input(queryIdx, trainIdx, now_kp, next_kp, camera_intrinsics, now_depth_img, now_img)
        training_input, new_uv = input_.insert_input()

        label_ = GENERATE_LABEL.generate_label(training_input, now_img.shape, camera_intrinsics)
        minimum_error, error_list = label_.get_label()

        # save training input and labels
        folder_name = output_folder + '/' + str(training_cnt)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # NEW uv is frpm previous frame
        #training_uv = [np.array(data_col[0]) for data_col in training_input]
        training_uv = [np.array(data_col) for data_col in new_uv]
        training_XYZ = [np.array(data_col[1]) for data_col in training_input]
        training_diversity3d = [np.array(data_col[2]) for data_col in training_input]
        #training_diversity2d = [np.array(data_col[3]) for data_col in training_input]
        training_response = [np.array(data_col[4]) for data_col in training_input]
        training_size = [np.array(data_col[5]) for data_col in training_input]

        error_list = np.array(error_list)
        error_list = error_list.reshape(-1, 1)

        np.save(folder_name + '/' + 'input_uv.npy', training_uv)
        np.save(folder_name + '/' + 'input_XYZ.npy', training_XYZ)
        #np.save(folder_name + '/' + 'input_diversity2d.npy', training_diversity2d)
        np.save(folder_name + '/' + 'input_diversity3d.npy', training_diversity3d)
        np.save(folder_name + '/' + 'input_response.npy', training_response)
        np.save(folder_name + '/' + 'input_size.npy', training_size)
        np.save(folder_name + '/' + 'error.npy', error_list)