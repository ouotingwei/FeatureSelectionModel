import numpy as np
import matplotlib.pyplot as plt
import feature_match as fm
import cv2 as cv

def draw_keypoints(image_path, keypoints, descriptor):
    # 讀取圖像
    image = plt.imread(image_path)
    print(descriptor[0])
    # 在圖像上繪製特徵點
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        size = int(kp[2])
        plt.scatter(x, y, s=size, color='red', alpha=0.5)
    
    # 顯示圖像
    plt.imshow(image)
    plt.title('Image with Keypoints')
    plt.show()

def convert_to_cv_keypoints(keypoints):
    cv_keypoints = []
    for kp in keypoints:
        x, y = kp[0], kp[1]
        size = kp[2]
        angle = kp[3]
        cv_kp = cv.KeyPoint(x, y, size, angle)
        cv_keypoints.append(cv_kp)
    return cv_keypoints

if __name__ == "__main__":
    image1_path = "/home/wei/deep_feature_selection/data/test_img/test1/1563004641.130956.png"  # 替換為你的圖像路徑
    keypoints1_path = "/home/wei/deep_feature_selection/data/test_img/test1/orb+boost/1563004641.130956 (1).png/keypoints.npy"  # 替換為保存特徵點的npy檔案路徑
    descriptor1_path = "/home/wei/deep_feature_selection/data/test_img/test1/orb+boost/1563004641.130956 (1).png/descriptors.npy"

    image2_path = "/home/wei/deep_feature_selection/data/test_img/test1/1563004641.164202.png"  # 替換為你的圖像路徑
    keypoints2_path = "/home/wei/deep_feature_selection/data/test_img/test1/orb+boost/1563004641.164202 (1).png/keypoints.npy"  # 替換為保存特徵點的npy檔案路徑
    descriptor2_path = "/home/wei/deep_feature_selection/data/test_img/test1/orb+boost/1563004641.164202 (1).png/descriptors.npy"
    
    keypoint1 = np.load(keypoints1_path)
    keypoint1 = convert_to_cv_keypoints(keypoint1)
    descriptor1 = np.load(descriptor1_path)
    img1 = cv.imread( "/home/wei/deep_feature_selection/data/test_img/test1/1563004641.130956.png" )


    keypoint2 = np.load(keypoints2_path)
    keypoint2 = convert_to_cv_keypoints(keypoint2)
    print(keypoint1[0])
    print(keypoint1[1])
    descriptor2 = np.load(descriptor2_path)
    img2 = cv.imread( "/home/wei/deep_feature_selection/data/test_img/test1/1563004641.164202.png" )

    #draw_keypoints(image1_path, keypoints, descriptor)
    # matching the feature between two  image frames
    matcher = fm.feature_match( img1, img2, keypoint1, keypoint2, descriptor1, descriptor2) 
    good_match = matcher.frame_match()
