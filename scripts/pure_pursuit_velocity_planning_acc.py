#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import time
import rospy
import rospkg
from math import cos,sin,pi ,sqrt,pow,atan2
from std_msgs.msg import Int32, Float32
from geometry_msgs.msg import Point,PoseWithCovarianceStamped
from nav_msgs.msg import Odometry,Path
from morai_msgs.msg import CtrlCmd, EgoVehicleStatus
from visualization_msgs.msg import Marker
import numpy as np
import time
import csv
import tf
from tf.transformations import euler_from_quaternion,quaternion_from_euler
from beginner_tutorials.msg import TrackingArray, TrackingPoint

# advanced_purepursuit 은 차량의 차량의 종 횡 방향 제어 예제입니다.
# Purpusuit 알고리즘의 Look Ahead Distance 값을 속도에 비례하여 가변 값으로 만들어 횡 방향 주행 성능을 올립니다.
# 횡방향 제어 입력은 주행할 Local Path (지역경로) 와 차량의 상태 정보 Odometry 를 받아 차량을 제어 합니다.
# 종방향 제어 입력은 목표 속도를 지정 한뒤 목표 속도에 도달하기 위한 Throttle control 을 합니다.
# 종방향 제어 입력은 longlCmdType 1(Throttle control) 이용합니다.

# 노드 실행 순서 
# 1. subscriber, publisher 선언
# 2. 속도 비례 Look Ahead Distance 값 설정
# 3. 좌표 변환 행렬 생성
# 4. Steering 각도 계산
# 5. PID 제어 생성
# 6. 도로의 곡률 계산
# 7. 곡률 기반 속도 계획
# 8. 제어입력 메세지 Publish

SAFE = 102
DECEL = 101
AEB = 100

class Vehicle:
    def __init__(self, position, velocity, acceleration, roll=0.0, pitch=0.0, yaw=0.0, yawrate=0.0):
        self.position = np.array(position)  # [e, n]
        self.velocity = np.array(velocity)  # [vx, vy]
        self.acceleration = np.array(acceleration)
        self.roll = roll
        self.yaw = yaw
        self.pitch = pitch
        self.yawrate = yawrate

class AdaptiveCruiseControl:
    def __init__(self, h=1.2, lambda_value=0.5, vehicle_length=4.0, max_acceleration=4.0, max_deceleration=-4.0):
        self.h = h  # Time gap
        self.lambda_value = lambda_value  # Lambda, 오차 수렴 속도를 조절하는 값
        self.vehicle_length = vehicle_length  # 차량 길이
        self.max_acceleration = max_acceleration
        self.max_deceleration = max_deceleration
        self.desired_distance = 0

    def calculate_spacing_error(self, current_distance, ego_speed):
        """
        Spacing error 계산
        """
        l_des = self.vehicle_length + self.h * ego_speed
        epsilon_i = - current_distance + l_des
        return epsilon_i

    def adjust_for_cut_in(self, current_distance, ego_speed, lead_speed):
        """
        끼어들기 상황을 고려한 목표 거리 축소 조정
        """
        # 기본 CTG 거리 계산
        l_des = self.vehicle_length + self.h * ego_speed

        # 끼어들기 상황 감지: 현재 거리 < CTG 거리
        if current_distance < l_des:
            # 상대속도에 따라 목표 거리 축소 조정
            relative_speed = ego_speed - lead_speed
            adjustment_factor = max(0.8, 1 - relative_speed / max(ego_speed, 0.1))  # 목표 거리 축소 비율 설정
            adjusted_l_des = l_des * adjustment_factor
        else:
            # 끼어들기 상황이 아니면 기본 CTG 거리 유지
            adjusted_l_des = l_des

        return adjusted_l_des

    def control_acceleration(self, ego_speed, lead_speed, current_distance):
        """
        주어진 식에 기반하여 가속도를 계산.
        """
        # 끼어들기 상황을 고려한 목표 거리 계산
        adjusted_distance = self.adjust_for_cut_in(current_distance, ego_speed, lead_speed)
        self.desired_distance = adjusted_distance
        # Spacing error와 delta_i 계산
        epsilon_i = - current_distance + adjusted_distance
        delta_i = epsilon_i + self.h * ego_speed
        
        # 가속도 계산
        relative_velocity = ego_speed - lead_speed
        epsilon_dot = relative_velocity
        acceleration = -(1 / self.h) * (epsilon_dot + self.lambda_value * delta_i)

        # 최대 가속도/감속도 제한
        acceleration = np.clip(acceleration, self.max_deceleration, self.max_acceleration)

        return acceleration

    
class pure_pursuit :
    def __init__(self):
        rospy.init_node('pure_pursuit', anonymous=True)

        #TODO: (1) subscriber, publisher 선언
        #rospy.Subscriber("/global_path", Path, self.global_path_callback)
        
        #rospy.Subscriber("/lattice_path", Path, self.path_callback)
        rospy.Subscriber("/local_path",Path,self.path_callback)

        rospy.Subscriber("/min_collision_time", Float32, self.min_collision_time_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/Ego_topic",EgoVehicleStatus, self.status_callback)
        #rospy.Subscriber("/distance_to_ego",Float32,self.distance_to_ego_callback)
        self.object_info_sub = rospy.Subscriber("/object_tracking", TrackingPoint, self.object_info_callback)
        self.ego_info_sub = rospy.Subscriber("/ego_tracking", TrackingPoint, self.ego_info_callback)
        self.intention_time = rospy.Subscriber("/intention_time", Float32, self.intention_time_callback)

        self.ctrl_cmd_pub = rospy.Publisher('ctrl_cmd',CtrlCmd, queue_size=1)

        self.status_marker_pub = rospy.Publisher('/status_marker', Marker, queue_size=1)

        self.ctrl_cmd_msg = CtrlCmd()
        self.ctrl_cmd_msg.longlCmdType = 2

        self.is_path = False
        self.is_odom = False 
        self.is_status = False
        #self.is_global_path = False
        self.is_TTC = False
        self.is_distance_to_ego = False

        self.is_look_forward_point = False

        self.forward_point = Point()
        self.current_postion = Point()

        self.vehicle_length = 2.84 #차체 길이 (임시 지정)
        self.lfd = 15
        if self.vehicle_length is None or self.lfd is None:
            print("you need to change values at line 57~58 ,  self.vegicle_length , lfd")
            exit()
        self.min_lfd = 10
        self.max_lfd = 30
        self.lfd_gain =  1.2 #0.78
        self.target_velocity = 50 # [km/h] Target 속도

        self.TTC = 999
        self.distance_to_ego = 999999
        self.max_deceleration = -9.8
        self.TTC_warning = 2.7
        self.TTC_AEB = 100

        self.system_status = SAFE
        self.previous_status = SAFE

        self.AEB_counter = 0
        self.decel_count = 0

        self.pid = pidControl()

        #acc_system = AdaptiveCruiseControl(time_gap=1.5, vehicle_length=4.0)
        acc_system = AdaptiveCruiseControl(h=1.2, lambda_value=0.4, vehicle_length=4.0)

        # # 차선 정보 불러오기
        # self.path = '/home/taewook/catkin_ws/src/beginner_tutorials/path/lane2.txt'
        # self.lane_data = self.load_lane_data(self.path)

        self.TTC_detect = 999
        

        rate = rospy.Rate(20) # 30hz
        while not rospy.is_shutdown():
            if self.is_path == True and self.is_odom == True and self.is_status == True:
                prev_time = time.time()
                
                # 스티어링 제어
                #self.current_waypoint = self.get_current_waypoint(self.status_msg,self.global_path)
                #self.target_velocity = self.velocity_list[self.current_waypoint]*3.6
                
                ego_speed = np.sqrt(self.ego_vehicle.velocity[0]**2 + self.ego_vehicle.velocity[1]**2)
                ego_unit_vector = np.array([
                    self.ego_vehicle.velocity[0] / max(ego_speed, 0.1),
                    self.ego_vehicle.velocity[1] / max(ego_speed, 0.1)
                ])
                
                position_difference = np.array([
                        self.lead_vehicle.position[0] - self.ego_vehicle.position[0],
                        self.lead_vehicle.position[1] - self.ego_vehicle.position[1]
                    ])
                

                # 위치 차이 벡터와 ego 방향 단위 벡터의 내적을 통해 current_distance 계산
                self.distance_to_ego = np.linalg.norm(position_difference)

                # Ego 차량의 진행 방향 단위 벡터
                ego_direction = self.ego_vehicle.velocity / max(ego_speed, 0.1)

                # 위치 차이 벡터 단위 벡터화
                position_direction = position_difference / max(self.distance_to_ego, 0.1)

                # 각도 계산 (라디안 -> 도)
                self.ego_surround_angle = np.degrees(np.arccos(np.clip(np.dot(ego_direction, position_direction), -1.0, 1.0)))

                steering = self.calc_pure_pursuit()
                if self.is_look_forward_point :
                    self.ctrl_cmd_msg.steering = steering
                else : 
                    rospy.loginfo("no found forward point")
                    self.ctrl_cmd_msg.steering = 0.0
                
                # 상태 머신
                # if not self.is_TTC: # 상대 차량이 사라졌을 때 TTC 초기화
                #     self.TTC = 999

                self.TTC_AEB = self.status_msg.velocity.x / (abs(self.max_deceleration)-1)  # velocity는 상대속도로 바꿔야 함
                self.system_status = self.state_machine(self.system_status, self.status_msg.velocity.x, self.TTC, self.TTC_warning, self.TTC_AEB)
                self.publish_status_marker()

                # 상태머신에 따른 속도 제어
                if self.system_status == SAFE:
                    self.ctrl_cmd_msg.longlCmdType = 2
                    self.ctrl_cmd_msg.velocity = self.target_velocity # km/h로 들어감
                
                elif self.system_status == DECEL:
                    self.ctrl_cmd_msg.longlCmdType = 3
                    

                    # 선행 차량 속도를 ego 차량의 진행 방향으로 투영하여 x, y 성분으로 분리
                    lead_speed_x = (self.lead_vehicle.velocity[0] * ego_unit_vector[0] +
                                    self.lead_vehicle.velocity[1] * ego_unit_vector[0])

                    lead_speed_y = (self.lead_vehicle.velocity[0] * ego_unit_vector[1] +
                                    self.lead_vehicle.velocity[1] * ego_unit_vector[1])

                    # x, y 성분을 가진 lead_speed_vector 생성
                    lead_speed_vector = np.array([lead_speed_x, lead_speed_y])
                    lead_speed = np.sqrt(lead_speed_vector[0]**2 + lead_speed_vector[1]**2) 
                    
                    # Ego 차량의 속도 벡터 및 단위 벡터 계산
                    ego_speed = np.sqrt(self.ego_vehicle.velocity[0]**2 + self.ego_vehicle.velocity[1]**2)
                    ego_direction_unit_vector = np.array([
                        self.ego_vehicle.velocity[0] / max(ego_speed, 0.1),
                        self.ego_vehicle.velocity[1] / max(ego_speed, 0.1)
                    ])
        
                    
                    # 가속도 계산
                    acc_distance = max(np.dot(position_difference, ego_direction_unit_vector), 0)
                    self.ctrl_cmd_msg.acceleration = acc_system.control_acceleration(ego_speed, lead_speed,acc_distance)
                    print(f"Acceleration: {self.ctrl_cmd_msg.acceleration}")
                    #acc_distance = max(np.dot(position_difference, ego_direction_unit_vector) -4, 0) # 차량 길이 4m 더하기
                    #self.ctrl_cmd_msg.acceleration = acc_system.control_acceleration(ego_speed, lead_speed, acc_distance)
                    
                    ## 기존 코드
                    # self.ctrl_cmd_msg.acceleration = -4
                    
                elif self.system_status == AEB:
                    self.ctrl_cmd_msg.longlCmdType = 3
                    self.ctrl_cmd_msg.acceleration = self.max_deceleration

                
                #output = self.pid.pid(self.target_velocity,self.status_msg.velocity.x*3.6)

                # if output > 0.0:
                #     self.ctrl_cmd_msg.accel = output
                #     self.ctrl_cmd_msg.brake = 0.0
                # else:
                #     self.ctrl_cmd_msg.accel = 0.0
                #     self.ctrl_cmd_msg.brake = -output

                # 충돌 감지 시 TTC 출력
                if (self.system_status == AEB or self.system_status == DECEL) and self.previous_status == 102:
                    self.TTC_detect = self.TTC

                #TODO: (8) 제어입력 메세지 Publish
                #rospy.loginfo("Type = {}".format(self.ctrl_cmd_msg.longlCmdType))
                if self.system_status == AEB:
                    rospy.loginfo("current status: {}".format("AEB!!!!!"))
                elif self.system_status == DECEL:
                    rospy.loginfo("current status: {}".format("ACC"))
                else:
                    rospy.loginfo("current status: {}".format("SAFE"))
                print("TTC: {}".format(self.TTC))
                print("Collision Detected: {} sec".format(self.TTC_detect))
                print("Distance: {}".format(self.distance_to_ego))
                print(f"Desired Distance: {acc_system.desired_distance}")
                print("Angle: {}".format(self.ego_surround_angle))
                print(f"counter: {self.decel_count}")
                print("----------------------------------------------")
                #print("steering: {}".format(steering))
                
                self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)
                self.previous_status = self.system_status
                
            rate.sleep()

    def publish_status_marker(self):
        marker = Marker()
        marker.header.frame_id = "Ego"  # Adjust frame if necessary
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.scale.z = 5.0  # Size of the text
        marker.pose.position.x = 68
        marker.pose.position.y = 22
        marker.pose.position.z = 0 # Position at the top-left corner; adjust as needed

        # Set text and color based on system status
        ljust_value = 23
        if self.system_status == SAFE:
            marker.text = "STATE: SAFE".ljust(ljust_value)
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
        elif self.system_status == DECEL:
            marker.text = "STATE: ACC".ljust(ljust_value)
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
        elif self.system_status == AEB:
            marker.text = "STATE: AEB".ljust(ljust_value)
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

        # Publish marker
        self.status_marker_pub.publish(marker)


    def load_lane_data(self, file_name):
        # 텍스트 파일에서 lane 데이터를 읽어옴
        lane_points = []
        full_path = file_name  # 경로와 파일명 결합
        with open(full_path, 'r') as file:
            reader = csv.reader(file, delimiter=' ')
            for row in reader:
                x, y, z = map(float, row)
                lane_points.append([x, y, z])
        return np.array(lane_points)  # NumPy 배열로 변환하여 반환
    
    def adaptive_cruise_control(self, ego_velocity, target_velocity, time_gap):
        """ACC 알고리즘을 통해 가속도를 계산"""
        # 시간 간격을 고려한 속도 차이
        distance_error = time_gap * target_velocity - ego_velocity
        # PID 제어기를 이용해 가속도를 계산
        acc_cmd = self.pid.pid(target_velocity, ego_velocity)
        return acc_cmd
    
    def path_callback(self,msg):
        self.is_path=True
        self.path=msg  

    def odom_callback(self,msg):
        self.is_odom=True
        odom_quaternion=(msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w)
        _,_,self.vehicle_yaw=euler_from_quaternion(odom_quaternion)
        self.current_postion.x=msg.pose.pose.position.x
        self.current_postion.y=msg.pose.pose.position.y

    def object_info_callback(self, msg):

        self.lead_vehicle = Vehicle(
            position=[msg.x, msg.y],
            velocity=[msg.vx, msg.vy],
            acceleration=[msg.ax, msg.ay],
            yaw=msg.yaw * np.pi/180, #rad/s
            yawrate = msg.yawrate * np.pi/180 #rad/s
            # roll = 0
            # pitch = 0
        )

    def ego_info_callback(self, msg):

        self.ego_vehicle = Vehicle(
            position = [msg.x, msg.y],
            velocity = [msg.vx, msg.vy],
            acceleration = [msg.ax, msg.ay],
            yaw = msg.yaw * np.pi/180, #rad/s 
            yawrate = msg.yawrate * np.pi/180 #rad/s
        )

    def status_callback(self,msg): ## Vehicle Status Subscriber 
        self.is_status=True
        self.status_msg=msg
    
    # def distance_to_ego_callback(self, msg):
    #     if msg.data:
    #         self.is_distance_to_ego = True
    #         self.distance_to_ego = msg.data  # msg.data로 접근
    #     else:
    #         self.distance_to_ego = 999999
    #         self.is_distance_to_ego = False
    
    # def global_path_callback(self,msg):
    #     self.global_path = msg
    #     self.is_global_path = True

    def min_collision_time_callback(self, msg):
    # 콜백 값이 정상적으로 들어왔을 때만 값 업데이트
        self.TTC = msg.data
        self.is_TTC = True
        # if msg.data:  # msg에 값이 있을 경우
        #     self.TTC = msg.data
        #     self.is_TTC = True
        # else:
        #     # 값이 없으면 기본값 999로 설정
        #     self.TTC = 999
        #     self.is_TTC = False

    def intention_time_callback(self, msg):
        self.intention_time = msg.data

    def calc_pure_pursuit(self,):

        #TODO: (2) 속도 비례 Look Ahead Distance 값 설정
        self.lfd = (self.status_msg.velocity.x) * self.lfd_gain
        
        if self.lfd < self.min_lfd : 
            self.lfd=self.min_lfd
        elif self.lfd > self.max_lfd :
            self.lfd=self.max_lfd
        rospy.loginfo(self.lfd)
        
        vehicle_position=self.current_postion
        self.is_look_forward_point= False

        translation = [vehicle_position.x, vehicle_position.y]

        #TODO: (3) 좌표 변환 행렬 생성
        trans_matrix = np.array([
                [cos(self.vehicle_yaw), -sin(self.vehicle_yaw),translation[0]],
                [sin(self.vehicle_yaw),cos(self.vehicle_yaw),translation[1]],
                [0                    ,0                    ,1            ]])

        det_trans_matrix = np.linalg.inv(trans_matrix)

        for num,i in enumerate(self.path.poses) :
            path_point=i.pose.position

            global_path_point = [path_point.x,path_point.y,1]
            local_path_point = det_trans_matrix.dot(global_path_point)    

            if local_path_point[0]>0 :
                dis = sqrt(pow(local_path_point[0],2)+pow(local_path_point[1],2))
                if dis >= self.lfd :
                    self.forward_point = path_point
                    self.is_look_forward_point = True
                    break
        
        #TODO: (4) Steering 각도 계산
        theta = atan2(local_path_point[1],local_path_point[0])
        steering = atan2((2*self.vehicle_length*sin(theta)),self.lfd)

        return steering

    def state_machine(self, current_status, ego_vx, TTC, TTC_warning, TTC_AEB):
        if current_status != DECEL:
            # 다른 상태로 전환되었을 때 decel_count를 초기화
            self.decel_count = 0
        
        if TTC < TTC_AEB and (current_status == SAFE or DECEL):
            return AEB
        
        #elif (TTC < TTC_warning or (self.intention_time>0.6 and self.distance_to_ego < 30)) and (current_status == SAFE):
        elif (TTC < TTC_warning) and (current_status == SAFE):
            return DECEL
        
        elif TTC > TTC_warning * 1.5 and (self.distance_to_ego > 20 or abs(self.ego_surround_angle) > 15) and current_status == DECEL:
            # 조건을 만족하면 decel_count 증가
            self.decel_count += 1
            #print(f"Decel count 증가: {self.decel_count}")

            # decel_count가 임계값을 넘으면 SAFE 상태로 전환
            if self.decel_count > 16:  # 임계값 예시
                self.decel_count = 0  # 초기화
                return SAFE
            else:
                return DECEL  # 조건이 아직 충족되지 않아 DECEL 유지

        elif TTC > TTC_warning and ego_vx < 1 and self.distance_to_ego > 6 and current_status == AEB:
            if self.AEB_counter >= 30*1: # 30Hz *1 초
                self.AEB_counter = 0
                self.distance_to_ego = 999999
                return SAFE
            else:
                self.AEB_counter+=1
                return AEB
                
        else:
            return current_status 

class pidControl:
    def __init__(self):
        self.p_gain = 0.3
        self.i_gain = 0.0001
        self.d_gain = 0.1
        self.prev_error = 0
        self.i_control = 0
        self.controlTime = 0.05

    def pid(self,target_vel, current_vel):
        error = target_vel - current_vel

        #TODO: (5) PID 제어 생성
        p_control = self.p_gain * error
        self.i_control += self.i_gain * error * self.controlTime
        d_control = self.d_gain * (error-self.prev_error) / self.controlTime

        output = p_control + self.i_control + d_control
        self.prev_error = error

        return output

if __name__ == '__main__':
    try:
        test_track=pure_pursuit()
    except rospy.ROSInterruptException:
        pass
