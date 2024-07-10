import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import training_input as TRAINING_INPUT
import orb_operation as ORB
import generate_label as GENERATE_LABEL

# dataset folder path( change to your own path )
color_file_path = '/media/ee605-wei/T7/資料集/OpenLORIS-Scene/201911-package/office1-1_7-package/office1-2/color'
depth_file_path = '/media/ee605-wei/T7/資料集/OpenLORIS-Scene/201911-package/office1-1_7-package/office1-2/aligned_depth'
camera_intrinsics = [611.4509887695312, 611.4857177734375, 433.2039794921875, 249.4730224609375] # fx, fy, cx, cy
output_folder = '/home/ee605-wei/FeatureSelectionModel/training_data/loris_800'
frameInterval = 1

if __name__ == '__main__':
    if not os.path.exists(color_file_path) or not os.path.exists(depth_file_path):
        print("file is not exist !")
        exit()

    # image path -> list
    data_preprocess = TRAINING_INPUT.data_preprocessing()
    color_img_list, depth_img_list, sequence_list =  data_preprocess.get_data_list(color_file_path, depth_file_path)

    num_of_features = []
    error = []

    for i in range(0, len(color_img_list), frameInterval):
        print(" [-] img : ", i)
        training_input = [] 

        # sequence t
        now_img = cv.imread( color_img_list[i] ) 
        now_depth_img = cv.imread( depth_img_list[i], cv.IMREAD_UNCHANGED )

        # sequence t+1
        next_img = cv.imread( color_img_list[i+frameInterval] )
        next_depth_img = cv.imread( depth_img_list[i+frameInterval], cv.IMREAD_UNCHANGED )

        # ORB Feature Extraction
        now_image_ = ORB.orb_features( now_img ) 
        now_kp, now_des = now_image_.feature_extract() 
        next_image_ = ORB.orb_features( next_img )
        next_kp, next_des = next_image_.feature_extract() 

        # matching the feature between two  image frames
        matcher_ = ORB.feature_match( now_img, next_img, now_kp, next_kp, now_des, next_des ) 
        queryIdx, trainIdx = matcher_.frame_match() # now->query, next->train

        '''
        Calculate the Visibility Loss
        '''
        match_list = np.ones(len(queryIdx))
        base_kp = []
        base_des = []
        for j in range(len(queryIdx)):
            base_kp.append(now_kp[queryIdx[j]])
            base_des.append(now_des[queryIdx[j]])

        base_des = cv.UMat(np.array(base_des))
        
        for j in range(i+frameInterval, i+40*frameInterval, frameInterval):
            # sequence t+1
            ne_img = cv.imread( color_img_list[i+frameInterval] )

            # ORB Feature Extraction
            ne_image_ = ORB.orb_features( ne_img )
            ne_kp, ne_des = ne_image_.feature_extract() 

            # matching the feature between two  image frames
            matcher_ = ORB.feature_match( now_img, ne_img, base_kp, ne_kp, base_des, ne_des ) 
            quIdx, trIdx = matcher_.frame_match() # now->query, next->train

            for k in range(len(quIdx)):
                match_list[quIdx[k]] += 1
        
        cnt = 0
        for j in range(len(match_list)):
            if match_list[j] == 40:
                cnt += 1
        print("visibility : ", cnt/len(match_list)*100, " %")

        

            
        '''
        Calculate the Reprojection Error Loss
        '''
        '''
        # sequence t
        now_img = cv.imread( color_img_list[i] ) 
        now_depth_img = cv.imread( depth_img_list[i], cv.IMREAD_UNCHANGED )

        # sequence t+1
        next_img = cv.imread( color_img_list[i+frameInterval] )
        next_depth_img = cv.imread( depth_img_list[i+frameInterval], cv.IMREAD_UNCHANGED )

        # ORB Feature Extraction
        now_image_ = ORB.orb_features( now_img ) 
        now_kp, now_des = now_image_.feature_extract() 
        next_image_ = ORB.orb_features( next_img )
        next_kp, next_des = next_image_.feature_extract() 

        # matching the feature between two  image frames
        matcher_ = ORB.feature_match( now_img, next_img, now_kp, next_kp, now_des, next_des ) 
        queryIdx, trainIdx = matcher_.frame_match() # now->query, next->train
        
        # insert features into training_input
        input_ = TRAINING_INPUT.set_training_input(queryIdx, trainIdx, now_kp, next_kp, camera_intrinsics, now_depth_img, now_img)
        training_input, new_uv = input_.insert_input()

        label_ = GENERATE_LABEL.generate_label(training_input, now_img.shape, camera_intrinsics)
        #minimum_error, error_list = label_.get_label()
        label_.pnp_solver_labeling()
        '''
        
        '''
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
        '''