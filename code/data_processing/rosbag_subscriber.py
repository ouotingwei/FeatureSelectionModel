 #!/usr/bin/env python
import rospy
import message_filters
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import pcl

def image_callback(msg):
    try:
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(1)
        
    except Exception as e:
        print(e)

def point_cloud_callback(msg):
    try:
        cloud = pcl.PointCloud()
        points = []
        for data in pcl.PointCloud.from_ros_msg(msg).to_list():
            points.append([data[0], data[1], data[2]])
        cloud.from_list(points)
    
        print("Received point cloud with %d points" % cloud.size)
        print("Point cloud header: ", msg.header)
        
    except Exception as e:
        print(e)

def gt_callback(msg):
    # todo
    return 0
def callback():
    #todo
    return 0


def filter():
    rospy.init_node('filter', anonymous=True)

    image = message_filters.Subscriber("/image", image_callback)
    pointcloud = message_filters.Subscriber("/pointcloud", point_cloud_callback)
    gt = message_filters.Subscriber("/gt", gt_callback)

    sync_listner = message_filters.TimeSynchronizer([image, pointcloud, gt], 10)

    sync_listner.registerCallback(callback)

if __name__ == '__main__':
    rospy.init_node('rosbag_subscriber', anonymous=True)

    rospy.Subscriber("/image_topic", Image, image_callback)

    rospy.spin()
