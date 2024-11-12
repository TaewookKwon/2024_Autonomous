#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import rospkg
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

class ReadPathsPub:
    def __init__(self):
        rospy.init_node('read_paths_pub', anonymous=True)
        self.path_pub = rospy.Publisher('/test_lanes', Path, queue_size=1)

        # 각각의 파일 경로 설정
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('beginner_tutorials')
        file_paths = [
            pkg_path + '/path/lane1.txt',
            pkg_path + '/path/lane2.txt',
            pkg_path + '/path/lane3.txt'
        ]

        # 하나의 Path 메시지에 모든 경로를 담을 것임
        self.global_path_msg = Path()
        self.global_path_msg.header.frame_id = '/map'

        # 각각의 파일에서 경로를 읽어 global_path_msg에 추가
        for file_path in file_paths:
            self.read_file(file_path)

        rate = rospy.Rate(20)  # 20Hz로 퍼블리시
        while not rospy.is_shutdown():
            # /test_lanes 토픽에 모든 경로 퍼블리시
            self.path_pub.publish(self.global_path_msg)
            rate.sleep()

    def read_file(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                tmp = line.split()
                read_pose = PoseStamped()
                read_pose.pose.position.x = float(tmp[0])
                read_pose.pose.position.y = float(tmp[1])
                read_pose.pose.position.z = float(tmp[2])  # z값 포함
                read_pose.pose.orientation.w = 1  # 회전은 기본값 설정
                self.global_path_msg.poses.append(read_pose)  # global_path_msg에 모든 경로 추가

if __name__ == '__main__':
    try:
        ReadPathsPub()
    except rospy.ROSInterruptException:
        pass
