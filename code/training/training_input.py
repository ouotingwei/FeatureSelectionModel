import numpy as np
from scipy.interpolate import interp1d
import os

import semantic as semantic

# inset Features into training input
# input vector
# [[ [u1(float), v1(float)], [X1(float), Y1(float), Z1(float)], [MEAN_DEPTH_AROUND_KP1(float) - kp_Z], [NUMBERS_OF_NEARBY_KP1(int)], [SEMANTIC_TYPE(int)] ],  ...]
def insert_input( match_2d_point, kp, intrinsic, depth_img, rgb_img ):
    training_input = []
    h = depth_img.shape[0]
    w = depth_img.shape[1]
    depth_scale = 1000

    #print(depth_img)

    for i in range( len(match_2d_point) ):
        #print( kp[match_2d_point[i]].pt)

        u = kp[match_2d_point[i]].pt[0] 
        v = kp[match_2d_point[i]].pt[1] 
        Z =  depth_img[int(v), int(u)]  / depth_scale
        X = (u - intrinsic[2]) * Z / intrinsic[0]
        Y = (v - intrinsic[3]) * Z / intrinsic[1]

        mean_depth_around_kp = get_mean_depth_aroung_kp( [u, v], Z, depth_img )
        depth_diff_around_kp = abs(mean_depth_around_kp - Z)

        num_of_nearby = get_numbers_of_nearby_kp( [u, v], kp, match_2d_point )

        u = u / w
        v = v / h

        training_input.append([ [u, v], [X, Y, Z], [depth_diff_around_kp], [num_of_nearby] ])

        #print([ [u, v], [X, Y, Z], [depth_diff_around_kp], [num_of_nearby] ])

    return training_input

def get_mean_depth_aroung_kp( kp_position, kp_Z, depth_img ):
    h = depth_img.shape[0]
    w = depth_img.shape[1]

    depth_scale = 1000

    mean_depth = None
    #[ u, v ] -> Z =  depth_img[int(v), int(u)]  / depth_scale
    # [1][2][3]
    # [4][5][6]
    # [7][8][9]
    rows, cols = np.indices((3, 3)) - 1

    row_indices = np.int32(kp_position[1]) + rows
    col_indices = np.int32(kp_position[0]) + cols

    # true -> indices is in the boundary(img)
    valid_indices = (row_indices >= 0) & (row_indices < depth_img.shape[0]) & (col_indices >= 0) & (col_indices < depth_img.shape[1])

    # calculate the mean_depth
    valid_depths = depth_img[row_indices[valid_indices], col_indices[valid_indices]]
    mean_depth = np.mean(valid_depths) / depth_scale if len(valid_depths) > 0 else None

    return mean_depth

def get_numbers_of_nearby_kp(kp_position, kp, match_2d_point):
    patch_w = 64
    patch_h = 48

    # kp range
    u_min = kp_position[0] - patch_w/2
    u_max = kp_position[0] + patch_w/2
    v_min = kp_position[1] - patch_h/2
    v_max = kp_position[1] + patch_h/2

    # num of kp in one patch
    cnt = sum(
        1 for i in match_2d_point
        if u_min < kp[i].pt[0] < u_max and v_min < kp[i].pt[1] < v_max
    )

    cnt -= 1
    return max(0, cnt) 

def find_gt_by_interpolation(sequence, gt_data):
    closet_index = None
    for i in range(len(gt_data)):
        if gt_data[i][0] > sequence:
            closet_index = i
            break

    big_index = closet_index
    small_index = closet_index - 1
    t_ = [gt_data[small_index][0], gt_data[big_index][0]]
    x_ = [gt_data[small_index][1], gt_data[big_index][1]]
    y_ = [gt_data[small_index][2], gt_data[big_index][2]]
    quaz_ = [gt_data[small_index][6], gt_data[big_index][6]]
    quaw_ = [gt_data[small_index][7], gt_data[big_index][7]]

    interp_x = interp1d(t_, x_, kind='linear')
    now_x = interp_x(sequence)

    interp_y = interp1d(t_, y_, kind='linear')
    now_y = interp_y(sequence)

    interp_quaz = interp1d(t_, quaz_, kind='linear')
    now_quaz = interp_quaz(sequence)

    interp_quaw = interp1d(t_, quaw_, kind='linear')
    now_quaw = interp_quaw(sequence)

    return [now_x, now_y, 0.0, 0.0, 0.0, now_quaz, now_quaw]

def get_data_list(color_file_path, depth_file_path, gt_file_path):
    '''
    input : color_file, depth_file, gt_file
    output : color_path_list, depth_path_list, gt_data_list
    '''
    print("Loading data ...")

    color_files = sorted(os.listdir(color_file_path), key=lambda x: float(os.path.splitext(x)[0]))
    depth_files = sorted(os.listdir(depth_file_path), key=lambda x: float(os.path.splitext(x)[0]))
    sequence = color_files

    gt_data = []
    with open(gt_file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            parts = line.strip().split(' ')
            time = float(parts[0])
            gt_entry = (time, *map(float, parts[1:]))
            gt_data.append(gt_entry)

    color_img_list = []
    depth_img_list = []
    gt_data_list = []
    sequence_list = []

    for i in range(len(color_files)):
        now_img = color_files[i].rsplit('.', 1)[0]
        now_depth = depth_files[i].rsplit('.', 1)[0]

        now_img_path = os.path.join(color_file_path, now_img + ".png")
        now_depth_path = os.path.join(depth_file_path, now_depth + ".png")

        color_img_list.append(now_img_path)
        depth_img_list.append(now_depth_path)

        sequence = float(now_img)
        sequence_list.append(sequence)
        gt_data_list.append( find_gt_by_interpolation(sequence, gt_data) )

    return color_img_list, depth_img_list, gt_data_list, sequence_list 