#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import CameraInfo

def camera_info_publisher():
    rospy.init_node('fake_camera_info_publisher')
    pub = rospy.Publisher('/camera/camera_info', CameraInfo, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz
    
    # CameraInfo 메시지 설정
    camera_info_msg = CameraInfo()
    camera_info_msg.header.frame_id = "Camera-3"
    camera_info_msg.height = 480
    camera_info_msg.width = 640
    
    # 계속해서 발행
    while not rospy.is_shutdown():
        camera_info_msg.header.stamp = rospy.Time.now()
        pub.publish(camera_info_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        camera_info_publisher()
    except rospy.ROSInterruptException:
        pass