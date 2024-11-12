#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from morai_msgs.msg  import EgoVehicleStatus
from geometry_msgs.msg import Point32,PoseStamped
from nav_msgs.msg import Odometry,Path
from beginner_tutorials.msg import TrackingPoint
import numpy as np
import matplotlib.pyplot as plt
import time
from math import sqrt, pow, cos, sin

class DataCollector():
    def __init__(self):
        rospy.init_node('DataCollector',anonymous=True)

        rospy.Subscriber("/global_path",Path, self.global_Path_callback)
        rospy.Subscriber('/ego_tracking', TrackingPoint, self.ego_info_callback)
    
        # 초기화
        self.global_path_msg=Path()
        self.global_path_msg.header.frame_id='/map'
        
        self.is_status=False
        self.is_path=False
        self.path_size=2 # 60
        self.path_rear_size=1 #10

        self.record = True

        self.x,self.y,self.vx,self.vy,self.heading = 0,0,0,0,0
        self.vector_x_left = [] # 왼쪽 차선의 x좌표 모음
        self.vector_y_left = [] # 왼쪽 차선의 y좌표 모음
        self.vector_x_right = [] #오른쪽 차선의 x좌표 모음
        self.vector_y_right =[] #오른쪽 차선의 y좌표 모음
        self.dr = 3.5 #차선 사이 간격 [m]

        rate = rospy.Rate(5) # 5hz
        while not rospy.is_shutdown():
            #rospy.loginfo(self.is_status)

            if self.is_status:
                self.x=self.ego_data.x
                self.y=self.ego_data.y
                self.vx=self.ego_data.vx
                self.vy=self.ego_data.vy
                self.heading=self.ego_data.yaw
                
                self.vector_x_left, self.vector_y_left, self.vector_x_right, self.vector_y_right = self.generate_lane_data(self.x,self.y,self.dr,np.pi/2)
                if self.record:
                    # rospy.loginfo('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(self.x,self.y,self.vx, self.vy, \
                    #     self.heading, self.vector_x_left, self.vector_y_left, self.vector_x_right, self.vector_y_right, self.dr))
                    print("-----------------------------\n")
                    rospy.loginfo("x: {}, y: {}".format(self.x,self.y))
                    rospy.loginfo("vx: {}, vy: {}".format(self.vx,self.vy))
                    rospy.loginfo("Heading: {}".format(self.heading))
                    rospy.loginfo("Left line X coordinate: {}".format(self.vector_x_left))
                    rospy.loginfo("Left line Y coordinate: {}".format(self.vector_y_left))
                    rospy.loginfo("Right line X coordinate: {}".format(self.vector_x_right))
                    rospy.loginfo("Right line Y coordinate: {}".format(self.vector_y_right))
            rate.sleep()
                
    def generate_lane_data(self,x,y,dr,rotate_angle):
        vector_x_left=[]
        vector_y_left=[]
        vector_x_right=[]
        vector_y_right=[]

        min_dis=float('inf')
        current_waypoint=-1
        for i,waypoint in enumerate(self.global_path_msg.poses) :

            distance=sqrt(pow(x-waypoint.pose.position.x,2)+pow(y-waypoint.pose.position.y,2))
            if distance < min_dis :
                min_dis=distance
                current_waypoint=i

        R=np.array([[cos(rotate_angle), -sin(rotate_angle)],[sin(rotate_angle), cos(rotate_angle)]])

        if current_waypoint != -1:
            if current_waypoint - self.path_rear_size>0 and current_waypoint - self.path_rear_size + self.path_size < len(self.global_path_msg.poses):
                start = current_waypoint - self.path_rear_size
                end = current_waypoint - self.path_rear_size + self.path_size
                offset = -1

            elif current_waypoint - self.path_rear_size<0:
                start = 0
                end = current_waypoint - self.path_rear_size + self.path_size
                offset = 1
            else:
                start = current_waypoint - self.path_rear_size
                end = len(self.global_path_msg.poses)
                offset = -1
                
            for num in range(start, end):
                # 방향벡터 구하기
                direction = np.array([self.global_path_msg.poses[num].pose.position.x-self.global_path_msg.poses[num+offset].pose.position.x, \
                                    self.global_path_msg.poses[num].pose.position.y-self.global_path_msg.poses[num+offset].pose.position.y])
                direction_normailize = direction/np.linalg.norm(direction)
                normal_normalize = np.dot(R,direction_normailize)
                
                vector_x_right.append(self.global_path_msg.poses[num].pose.position.x)
                vector_y_right.append(self.global_path_msg.poses[num].pose.position.y)
                vector_x_left.append(self.global_path_msg.poses[num].pose.position.x+normal_normalize[0]*dr)
                vector_y_left.append(self.global_path_msg.poses[num].pose.position.y+normal_normalize[1]*dr)
                 
            return vector_x_left, vector_y_left, vector_x_right, vector_y_right

    def ego_info_callback(self,msg):
        self.is_status=True
        self.ego_data = msg   

    def global_Path_callback(self,msg):
        self.global_path_msg = msg 

if __name__ == '__main__':
    try:
        data_collect=DataCollector()
    except rospy.ROSInterruptException:
        pass