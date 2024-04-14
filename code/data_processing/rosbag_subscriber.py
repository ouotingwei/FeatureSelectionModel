#!/usr/bin/env python

import rospy
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import message_filters
import numpy as np

img_cnt = 0
depth_image_cnt = 0
gt_cnt = 0


def depth_image_callback(msg):
    global depth_img
    depth_img = msg

    print("depth_img")

def gt_callback(msg):
    global gt_data

    for transform in msg.transforms:
        gt_data = np.array([transform.transform.translation.x,
                            transform.transform.translation.y,
                            transform.transform.translation.z,
                            transform.transform.rotation.x,
                            transform.transform.rotation.y,
                            transform.transform.rotation.z,
                            transform.transform.rotation.w])
    
    print("gt_data")
        
def camera_info_callback(msg):
    global camera_info
    camera_info = np.array([msg.height, msg.width, msg.K[0], msg.K[2], msg.K[4], msg.K[5]])

    print("camera_info")

def image_callback(msg):
    global img
    img = msg

    print("img")


def filter():
    rospy.init_node('rosbag_subscriber', anonymous=True)

    rospy.Subscriber("gt", TFMessage, gt_callback)
    rospy.Subscriber("/d400/aligned_depth_to_color/image_raw", Image, depth_image_callback)
    rospy.Subscriber("/d400/color/image_raw", Image, image_callback)
    rospy.Subscriber("/d400/color/image_raw", Image, image_callback)
    rospy.Subscriber("/d400/aligned_depth_to_color/camera_info", CameraInfo, camera_info_callback)


    rospy.spin()


if __name__ == '__main__':
    filter()
