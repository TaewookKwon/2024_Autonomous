#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from morai_msgs.msg  import EgoVehicleStatus,ObjectStatusList
from geometry_msgs.msg import Point32,PoseStamped
from nav_msgs.msg import Odometry,Path
from beginner_tutorials.msg import TrackingPoint
import numpy as np
import matplotlib.pyplot as plt
import time
from math import pi
from tf.transformations import euler_from_quaternion

# 최대범위 설정
max_yaw_rate_abs = 45

class ObjectInfo:
    def __init__(self):
        rospy.init_node('object_info', anonymous=True)

        # (1) subscriber, publisher 선언
        rospy.Subscriber("/Ego_topic",EgoVehicleStatus, self.status_callback)
        rospy.Subscriber("/Object_topic", ObjectStatusList, self.object_callback)
        #rospy.Subscriber("/odom", Odometry, self.odom_callback)

        self.object_tracking_pub = rospy.Publisher('/object_tracking', TrackingPoint, queue_size=100)
        self.ego_tracking_pub = rospy.Publisher('/ego_tracking', TrackingPoint, queue_size=100)

        # self.ebt_path_pub = rospy.Publisher('/ebt_path',Path, queue_size=1)
        # self.lattice_path_pub = rospy.Publisher('/target_velocity', int, queue_size = 1)

        # # 추가: is_crash 퍼블리셔 선언
        # self.is_crash_pub = rospy.Publisher('/is_crash', Bool, queue_size=1)

        # self.is_path = False
        self.is_status = False
        self.is_obj = False

        # self.ebt_path_size=20

        # self.dt = 0.1 # 0.1s
        

        #theta_offset= -88.65870666503906 * pi/180

        # 값 초기화
        yaw_previous = 0
        yaw_ebt = 0
        yaw_rate = 0
        ego_yaw_previous = None

        hz = 50
        #previous_time = time.time()
        rate = rospy.Rate(hz) # 30hz
        while not rospy.is_shutdown():
            if self.is_status:
                # print(f"Ego Heading: {self.ego_data.heading}")
                # print(f"Ego Velocity: {self.ego_data.velocity}")
                # print(f"Ego Acceleration: {self.ego_data.acceleration}")
                # print(f"Ego Position: {self.ego_data.position}")
                # print()
                #yaw_ego = (-self.ego_data.heading + theta_offset * 180/pi)%360
                current_time = rospy.Time.now()
                ego_transformed_vel, ego_transformed_acc = self.transform_coordinates(self.ego_data.velocity.x, self.ego_data.velocity.y, \
                    self.ego_data.acceleration.x, self.ego_data.acceleration.y, self.ego_data.heading)

                ego_tracking_msg = TrackingPoint()
                ego_tracking_msg.time = current_time
                ego_tracking_msg.x = self.ego_data.position.x
                ego_tracking_msg.y = self.ego_data.position.y
                ego_tracking_msg.vx = ego_transformed_vel[0]
                ego_tracking_msg.vy = ego_transformed_vel[1]
                ego_tracking_msg.ax = ego_transformed_acc[0]
                ego_tracking_msg.ay = ego_transformed_acc[1]
                ego_tracking_msg.yaw = self.ego_data.heading
                if ego_yaw_previous is not None:
                    ego_tracking_msg.yawrate = ((self.ego_data.heading)-ego_yaw_previous) / (1/hz)
                else: # 이전 yaw 값을 못 받아옴
                    ego_tracking_msg.yawrate = 0

                self.ego_tracking_pub.publish(ego_tracking_msg)

                ego_yaw_previous = self.ego_data.heading
                #print(f"Ego Heading: {ego_tracking_msg.yaw}")
                #print(f"Ego Yaw rate: {ego_tracking_msg.yawrate}")
                print("Map X: {}".format(ego_tracking_msg.x))
                print("Map Y: {}".format(ego_tracking_msg.y))
                # print("Map vx: {}".format(ego_tracking_msg.vx))
                # print("Map vy: {}".format(ego_tracking_msg.vy))
                print("Ego vx: {}".format(self.ego_data.velocity.x))
                print("Ego vx: {}".format(self.ego_data.velocity.y))
                print("Ego ax: {}".format(self.ego_data.acceleration.x))
                print("Ego ay: {}".format(self.ego_data.acceleration.y))
                print("Ego yaw rate: {}".format(ego_tracking_msg.yawrate))

                if self.is_obj:
                    for ebt_data in self.object_data.pedestrian_list:
                        if ebt_data.name == 'NCAP_EBT':
                            #rospy.loginfo(f"Found EBT1 pedestrian: {ebt}")
                            
                            #current_time = time.time()
                            yaw_ebt = ebt_data.heading
                            yaw_rate = (yaw_ebt-yaw_previous)/(1/hz) # [deg/s], 20Hz
                            yaw_rate = min(max_yaw_rate_abs,max(yaw_rate,-max_yaw_rate_abs)) # saturation 설정
                            yaw_previous = yaw_ebt

                            ebt_transformed_vel, ebt_transformed_acc = self.transform_coordinates(ebt_data.velocity.x/3.6, ebt_data.velocity.y/3.6,\
                                                                                         ebt_data.acceleration.x/3.6, ebt_data.acceleration.y/3.6, ebt_data.heading)

                            # EBT 값 확인
                            # print(f"EBT yaw rate: {yaw_rate}") #yaw rate 출력
                            # #print(f"EBT Yaw rate: {ebt_data.object_yaw_rate}")
                            #print(f"EBT Heading: {yaw_ebt}")
                            #print(f"Ego Heading: {self.ego_data.heading}")
                            # print("EBT Velocity:")
                            # print(f"{ebt_data.velocity}")
                            # print("EBT Acceleration:")
                            # print(f"{ebt_data.acceleration}")
                            # print("EBT Size:")
                            # print(f"{ebt_data.size}")
                            # print("EBT Position:")
                            # print(f"{ebt_data.position}")

                            # 메시지 생성 및 퍼블리시
                            ebt_tracking_msg = TrackingPoint()
                            ebt_tracking_msg.time = current_time
                            ebt_tracking_msg.x = ebt_data.position.x
                            ebt_tracking_msg.y = ebt_data.position.y
                            ebt_tracking_msg.vx = ebt_transformed_vel[0]
                            ebt_tracking_msg.vy = ebt_transformed_vel[1]
                            ebt_tracking_msg.ax = ebt_transformed_acc[0]
                            ebt_tracking_msg.ay = ebt_transformed_acc[1]
                            ebt_tracking_msg.yaw = ebt_data.heading
                            ebt_tracking_msg.yawrate = yaw_rate
                            #ebt_tracking_msg.yaw = yaw_ebt

                            self.object_tracking_pub.publish(ebt_tracking_msg)

                    # ebt_path_msg=Path()
                    # ebt_path_msg.header.frame_id='/map'

                    

                    #self.ebt_cv = self.cv_model(ebt_data.position.x, ebt_data.velocity.x, ebt_data.position.y, ebt_data.velocity.y, self.dt, self.ebt_path_size)
                    # self.ebt_cv=None
                    # self.ebt_cv = self.cv_model(ebt_data.position.x, ebt_transformed_vel[0], ebt_data.position.y, ebt_transformed_vel[1], self.dt, self.ebt_path_size)

                    # if self.ebt_cv is not None:
                    #     for num in range(len(self.ebt_cv)):
                    #         tmp_pose=PoseStamped()
                    #         tmp_pose.pose.position.x=self.ebt_cv[num][0]
                    #         tmp_pose.pose.position.y=self.ebt_cv[num][1]
                    #         tmp_pose.pose.orientation.w=1
                    #         ebt_path_msg.poses.append(tmp_pose)

                    #     self.ebt_path_pub.publish(ebt_path_msg)
                
                        
                # if self.checkObject(self.local_path, self.object_data):
                #     lattice_path = self.latticePlanner(self.local_path, self.status_msg)
                #     lattice_path_index = self.collision_check(self.object_data, lattice_path)

                #     # (7)  lattice 경로 메세지 Publish
                #     self.lattice_path_pub.publish(lattice_path[lattice_path_index])
                # else:
                #     self.lattice_path_pub.publish(self.local_path)
            rate.sleep()


    # def object_path_prediction(self):
    #     cv_pred = cv_model(x0, vx0, y0, vy0, dt,n_steps)
    #     ctrv_pred = ctrv_model(x0,vx0,y0,vy0,omega0,dt,n_steps)
    #     ca_pred = ca_model(x0, vx0, ax0, y0, vy0, ay0, dt,n_steps)

    def status_callback(self,msg): ## Vehicle Status Subscriber 
        self.is_status = True
        self.ego_data = msg

    def object_callback(self,msg):
        self.is_obj = True
        self.object_data = msg

    def transform_coordinates(self, vx, vy, ax, ay, heading):
        # 2D 회전 행렬 정의 (theta_offset만큼 회전)
        cos_theta = np.cos(heading*pi/180)
        sin_theta = np.sin(heading*pi/180)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [+sin_theta, cos_theta]
        ])
        
        
        # 속도 변환 (v_x, v_y)
        velocity = np.array([vx, vy])
        transformed_velocity = rotation_matrix @ [velocity[0], velocity[1]]
        
        # 가속도 변환 (a_x, a_y)
        acceleration = np.array([ax, ay])
        transformed_acceleration = rotation_matrix @ [acceleration[0],acceleration[1]]

        
        return transformed_velocity, transformed_acceleration

    
    # def cv_model(self, x0, vx0, y0, vy0, dt,n_steps):
    #     # 상태 벡터 초기화
    #     state = np.array([x0, vx0, y0, vy0])
    #     prediction_path =[[x0,y0,0]]

    #     # 상태 전이 행렬 A 정의
    #     A = np.array([
    #         [1, dt, 0, 0],
    #         [0, 1, 0, 0],
    #         [0, 0, 1, dt],
    #         [0, 0, 0, 1]
    #     ])


    #     # 시뮬레이션 실행
    #     for i in range(n_steps):
    #         # 다음 상태 계산
    #         state = A @ state
    #         prediction_path.append([state[0], state[2], dt*(i+1)])
        
    #     return prediction_path
        
    # def ctrv_model(self, x0,vx0,y0,vy0,omega0,dt,n_steps):
    #     # 상태 벡터 초기화
    #     state = np.array([x0, vx0, y0, vy0, omega0])
    #     prediction_path =[[x0,y0,0]]

    #     # # 상태 전이 행렬 A 정의
    #     # A = np.array([
    #     #     [1, np.sin(omega0*dt)/omega0, 0, -(1-np.cos(omega0*dt))/omega0, 0],
    #     #     [0, np.cos(omega0*dt), 0, -np.sin(omega0*dt), 0],
    #     #     [1, (1-np.cos(omega0*dt))/omega0, 0, np.sin(omega0*dt)/omega0, 0],
    #     #     [0, np.sin(omega0*dt), 0, np.cos(omega0*dt), 0],
    #     #     [0, 0, 0, 0, 1]
    #     # ])


    #     # 시뮬레이션 실행
    #     for i in range(n_steps):
    #         # 현재 상태로부터 다음 상태 계산
    #         x, vx, y, vy, omega = state
            
    #         # 위치 및 속도 업데이트
    #         if omega == 0:  # 각속도가 0일 경우, 직선 운동
    #             x_new = x + vx * dt
    #             y_new = y + vy * dt
    #             vx_new = vx
    #             vy_new = vy
    #         else:  # 각속도가 0이 아닌 경우, 회전 운동
    #             x_new = x + (vx * np.sin(omega * dt) + vy * (1 - np.cos(omega * dt))) / omega
    #             y_new = y + (vy * np.sin(omega * dt) - vx * (1 - np.cos(omega * dt))) / omega
                
    #             vx_new = vx * np.cos(omega * dt) - vy * np.sin(omega * dt)
    #             vy_new = vx * np.sin(omega * dt) + vy * np.cos(omega * dt)
            
    #         # 다음 상태 갱신
    #         state = np.array([x_new, vx_new, y_new, vy_new, omega])
    #         prediction_path.append([state[0], state[2], dt*(i+1)])

    #     return prediction_path

    #     # # 결과를 numpy 배열로 변환
    #     # results = np.array(results)

    #     # # 위치 데이터 추출
    #     # x_positions = results[:, 0]
    #     # y_positions = results[:, 2]

    #     # return x_positions,y_positions

    # def ca_model(self, x0, vx0, ax0, y0, vy0, ay0, dt,n_steps):
    #     # 상태 벡터 초기화
    #     state = np.array([x0, vx0, ax0, y0, vy0, ay0])
    #     prediction_path=[[x0,y0,0]]

    #     # 상태 전이 행렬 A 정의
    #     A = np.array([
    #         [1, dt, 0.5*dt**2, 0, 0, 0],
    #         [0, 1, dt, 0, 0, 0],
    #         [0, 0, 1, 0, 0, 0],
    #         [0, 0, 0, 1, dt, 0.5*dt**2],
    #         [0, 0, 0, 0, 1, dt],
    #         [0, 0, 0, 0, 0, 1]
    #     ])

    #     # 시뮬레이션 실행
    #     for i in range(n_steps):
    #         # 다음 상태 계산
    #         state = A @ state
    #         prediction_path.append([state[0], state[3], dt*(i+1)])

    #     return prediction_path

if __name__ == '__main__':
    try:
        ObjectInfo()
    except rospy.ROSInterruptException:
        pass
