import numpy as np
import cv2 as cv
import yaml

import semantic as semantic


# inset uvXYZ into training input
def insert_uvXYZ( training_input, match_2d_point, kp, intrinsic, depth_img, rgb_img ):
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

        u = u / w
        v = v / h

        training_input.append( [u, v, X, Y, Z] )
        #print([u, v, X, Y, Z])
    return training_input

# get sparity of the feature points

# get semantic info of the feature points