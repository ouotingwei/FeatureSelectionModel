import os
import cv2 as cv
import time

import training_input as TRAINING_INPUT
import orb_operation as ORB
import generate_label as GENERATE_LABEL

# dataset folder path( change to your own path )
color_file_path = '/media/ee605-wei/ROG_512_SSD/lab605_dynamic/color'
depth_file_path = '/media/ee605-wei/ROG_512_SSD/lab605_dynamic/depth'
tf_file_path = '/media/ee605-wei/ROG_512_SSD/lab605_dynamic/tf'
output_folder = '/home/ee605-wei/FeatureSelectionModel/training_data/dynamic_data'

camera_intrinsics = [589.662879, 592.577963, 313.100104, 251.210229] # fx, fy, cx, cy -> realsense D435 (605)
frameInterval = 1
VisibilityInterval = 3

if __name__ == '__main__':
    if not os.path.exists(color_file_path) or not os.path.exists(depth_file_path):
        print("file is not exist !")
        exit()

    # image path -> list
    data_preprocess = TRAINING_INPUT.data_preprocessing()
    color_img_list = data_preprocess.get_data_list(color_file_path, '.png')
    depth_img_list = data_preprocess.get_data_list(depth_file_path, '.png')
    tf_list = data_preprocess.get_data_list(tf_file_path, '.txt')

    num_of_features = []
    error = []
    cnt = 0

    for i in range(300, len(color_img_list), frameInterval): 
        training_input = [] 
        print(i)
        # sequence t
        now_img = cv.imread(color_img_list[i]) 
        now_depth_img = cv.imread(depth_img_list[i], cv.IMREAD_UNCHANGED)
        now_tf = data_preprocess.get_tf(tf_list[i])

        # sequence t+1
        if i + frameInterval < len(color_img_list):
            next_img = cv.imread(color_img_list[i+frameInterval])
            next_depth_img = cv.imread(depth_img_list[i+frameInterval], cv.IMREAD_UNCHANGED)
            next_tf = data_preprocess.get_tf(tf_list[i+frameInterval])
        else:
            continue 

        # ORB Feature Extraction
        now_image_ = ORB.orb_features(now_img) 
        now_kp, now_des = now_image_.feature_extract() 
        
        '''
        for i, kp in enumerate(now_kp):
            print(f"KeyPoint {i}:")
            print(f"  Angle: {kp.angle}")
            print(f"  Class_id: {kp.class_id}")
            print(f"  Octave: {kp.octave}")
            print(f"  Pt: {kp.pt}")
            print(f"  Response: {kp.response}")
            print(f"  Size: {kp.size}")
        '''
        
        next_image_ = ORB.orb_features(next_img)
        next_kp, next_des = next_image_.feature_extract() 

        # matching the feature between two  image frames
        matcher_ = ORB.feature_match(now_img, next_img, now_kp, next_kp, now_des, next_des) 
        queryIdx, trainIdx = matcher_.frame_match()  # now->query, next->train

        # calculate the transition matrix between now and next images.
        transition_matrix = data_preprocess.get_translation_matrix(now_tf, next_tf)

        # generate the points' input
        input_ = TRAINING_INPUT.set_training_input(queryIdx, trainIdx, now_kp, next_kp, camera_intrinsics, now_depth_img, now_img)
        point_3d_query, point_2d_train, octave = input_.get_point()

        #training_input, new_uv = input_.insert_input()

        # Calculate the Reprojection Error
        reprojectrion_label_ = GENERATE_LABEL.Reprojection(point_3d_query, point_2d_train, octave, next_img, camera_intrinsics, transition_matrix)
        error_list, inlier_rate = reprojectrion_label_.get_reprojection()

        # Calculate the Visibility
        visibility_label_ = GENERATE_LABEL.Visibility(color_img_list, i, now_img, now_kp, now_des, queryIdx, VisibilityInterval)
        visible_list, visible_rate = visibility_label_.get_visibility()
        
        if inlier_rate > 99:
            cnt += 1
            print(i)
            print("visible rate : ", visible_rate)
            print("inlier rate : ", inlier_rate)

        '''
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

    print(cnt)