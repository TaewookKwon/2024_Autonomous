#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from std_msgs.msg import Int32, Float32
from geometry_msgs.msg import Point32,PoseStamped
from nav_msgs.msg import Odometry
from beginner_tutorials.msg import TrajectoryArray, TrajectoryPoint, TrackingArray, TrackingPoint  # 실제 메시지 파일 경로를 맞게 수정
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import euler_from_quaternion
from scipy.spatial import KDTree
import tf
import time
import csv
import os
from math import pi
import torch
import torch.nn as nn
import pandas as pd
import copy


# Define the LSTM model
class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(TrajectoryLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)  # Move LSTM to the appropriate device
        self.fc = nn.Linear(hidden_size, output_size).to(device)  # Move Linear layer to the appropriate device
        self.device = device  # Store the device

    def forward(self, x, num_layers=2, hidden_size=512):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(self.device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(self.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])  # Use output from the last time step
        return out

    
class Vehicle:
    def __init__(self, position, velocity, acceleration, roll=0.0, pitch=0.0, yaw=0.0, yawrate=0.0):
        self.position = np.array(position)  # [e, n]
        self.velocity = np.array(velocity)  # [vx, vy]
        self.acceleration = np.array(acceleration)
        self.roll = roll
        self.yaw = yaw
        self.pitch = pitch
        self.yawrate = yawrate

class CollisionChecker:
    def __init__(self):
        self.vehicle = []
        self.ego_vehicle = []
        #self.current_road_option = 0
        self.ego_roll = 0.0 # [rad]
        self.ego_pitch = 0.0
        self.ego_yaw = 0.0
        self.min_collision_time = 999
        self.shared_time = None
        self.last_saved_time = None
        
        # ---------------------------- Save ego state ----------------------------- #
        self.ego_state_file = '/home/taewook/catkin_ws/src/beginner_tutorials/data/ego_state_data.csv'
        self.future_trajectory_file = '/home/taewook/catkin_ws/src/beginner_tutorials/data/future_trajectory.csv'

        self.init_csv_files()
        # ------------------------------------------------------------------------- #

        rospy.init_node('collision_checker', anonymous=True)

        # 이륜차 동역학을 고려하는 변수의 초기화
        self.prev_time = None
        self.roll = 0
        self.roll_rate = 0
        self.roll_accel = 0
        self.yaw_accel = 0
        self.prev_roll = None
        self.prev_roll_rate = 0
        
        self.h = 1.5  # Vehicle height in meters
        self.l_m = 2.5  # Distance between wheels in meters
        self.g = 9.81  # Gravity constant (m/s^2)

        # Subsciber 선언
        self.object_info_sub = rospy.Subscriber("/object_tracking", TrackingPoint, self.object_info_callback)
        self.ego_info_sub = rospy.Subscriber("/ego_tracking", TrackingPoint, self.ego_info_callback)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)

        
        # publisher 선언
        self.min_collision_time_pub = rospy.Publisher("min_collision_time", Float32, queue_size=10)
        self.trajectory_marker_pub = rospy.Publisher("trajectory_prediction", MarkerArray, queue_size=800)
        
        # 차선 데이터 불러오기
        self.path = '/home/taewook/catkin_ws/src/beginner_tutorials/path/'
        self.lane_files = ['lane1.txt', 'lane2.txt', 'lane3.txt']
        self.lane_data = [None, None, None] # 차선 데이터 (x값)
        self.lane_data_points = [self.load_lane_data(f) for f in self.lane_files]  # 경로와 파일명을 합쳐서 파일 로드
        
        # # LSTM 모델 초기화
        model_path = '/home/taewook/LSTM/trajectory_lstm_model_5.pth'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm_model = self.load_lstm_model(model_path)

        # 기존 초기화 코드
        self.data_buffer = []  # 데이터를 저장할 리스트
        self.time_step = 0.1   # 0.1초 간격
        self.max_steps = 30     # 최대 스텝 수 (3초)

        #ego 콜백에서 데이터를 받아오는 시간 기준
        self.last_update_time = 0

        # 평균, 표준편차 불러옴
        raw_file_path = '/home/taewook/catkin_ws/src/beginner_tutorials/data/ego_tracking_data_raw_combine.csv'
        raw_data = pd.read_csv(raw_file_path, header=None)  # 헤더가 없다고 가정
        self.data_mean = raw_data.mean().values[1:].tolist()  # 1부터 끝까지 선택하여 저장
        self.data_std = raw_data.std().values[1:].tolist()    # 1부터 끝까지 선택하여 저장

    def load_lstm_model(self, model_path):
        """LSTM 모델 로드하는 메서드."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TrajectoryLSTM(input_size=7, hidden_size=512, num_layers=2, output_size=90, device=device)  # 모델의 구조에 맞게 설정
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # 평가 모드로 설정
        return model
    
    def init_csv_files(self):
            if not os.path.exists(self.ego_state_file):
                self.write_csv_header(self.ego_state_file, ['time', 'x', 'y', 'vx', 'vy'])
            if not os.path.exists(self.future_trajectory_file):
                header = ['time'] + [f'x{i+1}' for i in range(30)] + [f'y{i+1}' for i in range(30)]
                self.write_csv_header(self.future_trajectory_file, header)

    def write_csv_header(self, file_path, header):
        try:
            with open(file_path, 'w', newline='') as f:
                csv.writer(f).writerow(header)
        except Exception as e:
            rospy.logwarn(f"CSV 파일 헤더 작성 실패: {e}")

    def load_lane_data(self, file_name):
        # 텍스트 파일에서 lane 데이터를 읽어옴
        lane_points = []
        full_path = self.path + file_name  # 경로와 파일명 결합
        with open(full_path, 'r') as file:
            reader = csv.reader(file, delimiter=' ')
            for row in reader:
                x, y, z = map(float, row)
                lane_points.append([x, y, z])
        return np.array(lane_points)  # NumPy 배열로 변환하여 반환

    
    def find_closest_point(self, lane, ego_x, ego_y):
        # KDTree를 사용해 가장 가까운 점을 빠르게 찾음
        tree = KDTree(lane[:, :2])  # (x, y) 좌표로 KDTree 생성
        dist, closest_index = tree.query([ego_x, ego_y])  # 가장 가까운 점 검색
        return lane[closest_index]  # 가장 가까운 점의 좌표 반환 (x, y, z)
    
    def object_info_callback(self, msg):

        self.vehicle = Vehicle(
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
        
        if hasattr(self, 'lane_data_points') and self.lane_data_points:
            # rospy.loginfo("lane point.........")
            # lane123
            # 각 차선에서 가장 가까운 좌표 찾기
            for i, lane in enumerate(self.lane_data_points):
                closest_point = self.find_closest_point(lane, self.ego_vehicle.position[0], self.ego_vehicle.position[1])
                self.lane_data[i] = closest_point[0]  # x 좌표만 저장

            current_time = time.time()
            
            # last_update_time이 없거나 초기화되지 않은 경우 기본값 설정
            if not hasattr(self, 'last_update_time'):
                self.last_update_time = current_time  # 초기값 설정

            if current_time - self.last_update_time >= 0.1:
                self.last_update_time = current_time
                # Only add data to data_buffer if it has been initialized
                if hasattr(self, 'data_buffer') :  # Check if data_buffer is defined and not empty
                    #rospy.loginfo("Start saving data...")
                    # history를 직렬화해서 data_buffer에 저장 : [x1, y1, yaw1, ... ,x30, y30, yaw30]
                    input_data = [self.ego_vehicle.position[0], self.ego_vehicle.position[1],
                                  np.sqrt(self.ego_vehicle.velocity[0]**2 + self.ego_vehicle.velocity[1]**2),
                                  self.ego_vehicle.yaw,self.lane_data[0], self.lane_data[1], self.lane_data[2]
                                ]
                    input_data_normalize = [(input_data[i]-self.data_mean[i]) / self.data_std[i] for i in range(7)]

                    self.data_buffer.extend(input_data_normalize)
                
                    # 데이터가 max_steps를 초과하면 가장 오래된 데이터 삭제
                    if len(self.data_buffer) > self.max_steps * 7:
                        self.data_buffer = self.data_buffer[-self.max_steps * 7:]  # 마지막 210개 데이터 유지
                    
                    # rospy.loginfo(f"Length!!!!! : {len(self.data_buffer)}")
                    # rospy.loginfo("lane1: {}".format(self.data_buffer[4]))
                    # rospy.loginfo("lane2: {}".format(self.data_buffer[5]))
                    # rospy.loginfo("lane3: {}".format(self.data_buffer[6]))

                    # rospy.loginfo("x value before 3s: {}".format(self.data_buffer[0]))
                    # rospy.loginfo("Y value before 3s: {}".format(self.data_buffer[1]))
                    # rospy.loginfo("v value before 3s: {}".format(self.data_buffer[2]))
                    # rospy.loginfo("YAW value before 3s: {}".format(self.data_buffer[3]))

        # 현재 시간을 가져옵니다.
        # current_time = msg.time.secs + msg.time.nsecs * 1e-9

        # 0.1초마다 데이터 저장 로직 실행
        # if current_time is not None:
        #     if self.last_saved_time is not None and (current_time - self.last_saved_time) >= 0.1:
        #         # Save ego state to CSV
        #         self.save_ego_state(current_time, msg.x, msg.y)

        #         # Predict future trajectory
        #         time_horizon = 3.0
        #         dt = 0.1
        #         # ego_trajectory_CTRV = self.predict_motion_CTRV(self.ego_vehicle, time_horizon, dt)
        #         # ego_trajectory_CTRA = self.predict_motion_CTRA(self.ego_vehicle, time_horizon, dt)
        #         # ego_trajectory_CV = self.predict_motion_CV(self.ego_vehicle, time_horizon, dt)

        #         # CTRV 예측
        #         ego_vehicle_copy = copy.deepcopy(self.ego_vehicle)  # 깊은 복사
        #         ego_trajectory_CTRV = self.predict_motion_CTRV(ego_vehicle_copy, time_horizon, dt)
        #         del ego_vehicle_copy  # 복사한 객체 삭제

        #         # CTRA 예측
        #         ego_vehicle_copy = copy.deepcopy(self.ego_vehicle)  # 깊은 복사
        #         ego_trajectory_CTRA = self.predict_motion_CTRA(ego_vehicle_copy, time_horizon, dt)
        #         del ego_vehicle_copy  # 복사한 객체 삭제

        #         # CV 예측
        #         ego_vehicle_copy = copy.deepcopy(self.ego_vehicle)  # 깊은 복사
        #         ego_trajectory_CV = self.predict_motion_CV(ego_vehicle_copy, time_horizon, dt)
        #         del ego_vehicle_copy  # 복사한 객체 삭제

        #         # Dynamics 고려
        #         ego_vehicle_copy = copy.deepcopy(self.ego_vehicle)  # 깊은 복사
        #         ego_trajectory_dynamics = self.predict_motion_dynamics(self.ego_vehicle, self.ego_roll, time_horizon, dt)
        #         del ego_vehicle_copy  # 복사한 객체 삭제

        #         # Save predicted trajectory to CSV
        #         x_values = [point.x for point in ego_trajectory_CTRV.points] + [point.x for point in ego_trajectory_CTRA.points] + [point.x for point in ego_trajectory_CV.points] + [point.x for point in ego_trajectory_dynamics.points]
        #         y_values = [point.y for point in ego_trajectory_CTRV.points] + [point.y for point in ego_trajectory_CTRA.points] + [point.y for point in ego_trajectory_CV.points] + [point.y for point in ego_trajectory_dynamics.points]
        #         self.save_future_trajectory(current_time, x_values, y_values)

        #         # 마지막 저장 시간을 업데이트합니다.
        #         self.last_saved_time = current_time
            
        #     elif self.last_saved_time is None:
        #         self.last_saved_time = current_time

    def odom_callback(self,msg):
        self.is_odom=True
        odom_quaternion=(msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w)
        temp,_,_=euler_from_quaternion(odom_quaternion)
        self.ego_roll = temp * 15
    
    # # Predict trajectory using LSTM
    # def predict_motion_LSTM(self, input_tensor, mean, std, dt):
    #     with torch.no_grad():
    #         predicted_trajectory = self.lstm_model(input_tensor)
        
    #     # 예측 결과를 (30, 3) 형태로 변환 (각 타임스텝의 [x, y, yaw])
    #     predicted_trajectory = predicted_trajectory.view(30, 3).cpu().numpy()

    #     data_mean_output = np.array([self.data_mean[i] for i in (0, 1, 3)])
    #     data_std_output = np.array([self.data_std[i] for i in (0, 1, 3)])

    #     #역정규화
    #     predicted_trajectory = predicted_trajectory * data_std_output + data_mean_output

    #     trajectory_array = TrajectoryArray()
        
    #     # curr_point = TrajectoryPoint()
    #     # curr_point.x = self.ego_vehicle.position[0]
    #     # curr_point.y = self.ego_vehicle.position[1]
    #     # curr_point.yaw = self.ego_vehicle.yaw
    #     # curr_point.time = 0.0
    #     # trajectory_array.points.append(curr_point)

    #     for t in range(0,30):
    #         point = TrajectoryPoint()
    #         point.x = predicted_trajectory[t, 0]  # 예측된 x 값
    #         point.y = predicted_trajectory[t, 1]  # 예측된 y 값
    #         point.yaw = predicted_trajectory[t, 2]  # 예측된 yaw 값
    #         point.time = t * 0.1  # 각 타임스텝의 시간 (0.1초 간격)

    #         trajectory_array.points.append(point)  # TrajectoryArray에 추가

    #     return trajectory_array

    # Predict trajectory using LSTM
    def predict_motion_LSTM(self, input_tensor, mean, std, dt):
        with torch.no_grad():
            predicted_trajectory = self.lstm_model(input_tensor)
        
        # 예측 결과를 (30, 3) 형태로 변환 (각 타임스텝의 [x, y, yaw])
        predicted_trajectory = predicted_trajectory.view(30, 3).cpu().numpy()

        data_mean_output = np.array([self.data_mean[i] for i in (0, 1, 3)])
        data_std_output = np.array([self.data_std[i] for i in (0, 1, 3)])

        #역정규화
        predicted_trajectory = predicted_trajectory * data_std_output + data_mean_output

        trajectory_array = TrajectoryArray()
        
        curr_point = TrajectoryPoint()
        curr_point.x = self.ego_vehicle.position[0]
        curr_point.y = self.ego_vehicle.position[1]
        curr_point.yaw = self.ego_vehicle.yaw
        curr_point.time = 0.0
        trajectory_array.points.append(curr_point)

        temp_x = self.ego_vehicle.position[0]
        temp_y = self.ego_vehicle.position[1]

        #offset_x = temp_x - predicted_trajectory[0, 0]
        #offset_y = temp_y - predicted_trajectory[0, 1]

        for t in range(0,30):
            point = TrajectoryPoint()
            point.x = predicted_trajectory[t, 0] + temp_x  # 예측된 x 값
            point.y = predicted_trajectory[t, 1] + temp_y # 예측된 y 값
            point.yaw = predicted_trajectory[t, 2]  # 예측된 yaw 값
            point.time = (t+1) * 0.1  # 각 타임스텝의 시간 (0.1초 간격)

            trajectory_array.points.append(point)  # TrajectoryArray에 추가

        return trajectory_array

    
    def calculate_circles_surround(self, position, yaw, vehicle_width, vehicle_length, radius):
        """
        Calculate the positions of 3 circles (front, center, rear) based on vehicle's position and yaw.
        
        Args:
        - position (numpy array of shape (2,)): [x, y] position of the vehicle
        - yaw (float): The yaw angle (in radians)
        - vehicle_width, vehicle_length, radius: Not used in this specific function
        
        Returns:
        - circles (list of numpy arrays): List containing the [x, y] coordinates of the front, center, and rear circles
        """
        circles = []

        # Vehicle's center position
        center = np.array([position[0], position[1]])

        # Define the rotation matrix using yaw (2D rotation)
        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], 
                                    [np.sin(yaw), np.cos(yaw)]])

        # Define points in the vehicle's local frame
        front_point = np.array([1.5, 0.0])  # x: +2, y: 0
        center_point = np.array([0.0, 0.0])  # x: 0, y: 0
        rear_point = np.array([-1.5, 0.0])  # x: -2, y: 0

        # Rotate the points to the global frame using the yaw value
        front_global = np.dot(rotation_matrix, front_point)
        center_global = np.dot(rotation_matrix, center_point)
        rear_global = np.dot(rotation_matrix, rear_point)

        # Translate the points to the vehicle's global position
        front_circle = center + front_global
        center_circle = center + center_global
        rear_circle = center + rear_global

        # Store the circle positions in the list
        circles.append(front_circle)
        circles.append(center_circle)
        circles.append(rear_circle)

        return circles


    def calculate_circles(self, position, yaw, vehicle_width, vehicle_length, radius):
        """
        Calculate the positions of 3 circles (front, center, rear) using roll, pitch, and yaw (RPY) angles.
        
        Args:
        - position (numpy array of shape (2,)): [x, y] position of the vehicle
        - roll, pitch, yaw (float): The roll, pitch, and yaw angles (in radians)
        - vehicle_width, vehicle_length, radius: Not used in this specific function
        
        Returns:
        - circles (list of numpy arrays): List containing the [x, y] coordinates of the front, center, and rear circles
        """
        circles = []

        # Vehicle's center position
        center = np.array([position[0], position[1]]) # map frame

        # Define the transformation matrix using roll, pitch, yaw (RPY)
        R = np.array([[np.cos(yaw), -np.sin(yaw)], 
                    [np.sin(yaw), np.cos(yaw)]])


        # Define points in the vehicle's local frame (assuming z=0 for a 2D transformation)
        front_point = np.array([2.84, 0.0])  # x: +2, y: 0
        center_point = np.array([1.34, 0.0])  # x: 0, y: 0
        rear_point = np.array([-0.16, 0.0])  # x: -2, y: 0

        # Transform the points to the global frame using the inverse transformation
        front_global = np.dot(R, front_point)
        center_global = np.dot(R, center_point)
        rear_global = np.dot(R, rear_point)

        # Convert the 3D points to 2D by ignoring the z component
        front_circle = center + np.array([front_global[0], front_global[1]])
        center_circle = center + np.array([center_global[0], center_global[1]])
        rear_circle = center + np.array([rear_global[0], rear_global[1]])

        # Store the circle positions in the list
        circles.append(front_circle)
        circles.append(center_circle)
        circles.append(rear_circle)

        return circles

    def visualize_circles(self, circle_positions, vehicle_id,ego_vehicle_radius, surround_vehicle_radius):
        """
        Visualize the circles at each position using MarkerArray.

        Args:
        - circle_positions: List of circle positions (x, y)
        - vehicle_id: Unique ID for the vehicle (0 for ego, 1 for surround)
        """
        marker_array = MarkerArray()

        # vehicle_id에 따른 반지름 설정
        if vehicle_id == 0:  # Ego vehicle
            radius = ego_vehicle_radius
        elif vehicle_id == 1:  # Surround vehicle
            radius = surround_vehicle_radius
        else:
            rospy.logwarn(f"Unknown vehicle_id: {vehicle_id}")
            return

        for i, circle_pos in enumerate(circle_positions):
            circle_marker = Marker()
            circle_marker.header.frame_id = "map"
            circle_marker.header.stamp = rospy.Time.now()
            circle_marker.ns = "vehicle_circles"
            circle_marker.action = Marker.ADD
            circle_marker.pose.orientation.w = 1.0
            circle_marker.lifetime = rospy.Duration(1.0)
            circle_marker.id = vehicle_id * 1000 + i  # Unique ID for each circle
            circle_marker.type = Marker.SPHERE
            
            # Set the scale (diameter is twice the radius)
            circle_marker.scale.x = radius * 2  # Set diameter for x-axis
            circle_marker.scale.y = radius * 2  # Set diameter for y-axis
            circle_marker.scale.z = 0.01  # Flat circles on the ground

            # Set the position of the circle
            circle_marker.pose.position.x = circle_pos[0]
            circle_marker.pose.position.y = circle_pos[1]
            circle_marker.pose.position.z = 0  # Flat on the ground

            # Set the color of the circle
            circle_marker.color.r = 1.0
            circle_marker.color.g = 0.8
            circle_marker.color.b = 0.0
            circle_marker.color.a = 0.5  # Semi-transparent

            marker_array.markers.append(circle_marker)


    def visualize_trajectory(self, trajectory, vehicle_id, ego_radius, surround_radius):
        """
        Visualize the trajectory with circles at each point.
        
        Args:
        - trajectory: The predicted trajectory of the vehicle (TrajectoryArray)
        - vehicle_id: Unique ID for the vehicle (0 for ego, 1 for surround)
        - ego_radius: Radius for ego vehicle circles (default = 1.0)
        - surround_radius: Radius for surround vehicle circles (default = 0.5)
        """
        #rospy.loginfo("visulaize trajectory......................................................")
        marker_array = MarkerArray()

        # Line strip for the vehicle's trajectory
        line_strip = Marker()
        line_strip.header.frame_id = "map"
        line_strip.header.stamp = rospy.Time.now()
        line_strip.ns = f"vehicle_{vehicle_id}_trajectory_lines"
        line_strip.action = Marker.ADD
        line_strip.pose.orientation.w = 1.0
        line_strip.lifetime = rospy.Duration(1.0)
        line_strip.id = vehicle_id  # Unique ID for each vehicle trajectory
        line_strip.type = Marker.LINE_STRIP
        line_strip.scale.x = 0.15  # Line thickness

        # Set the line color based on vehicle_id (0 for ego, 1 for surround)
        if vehicle_id == 0:  # Ego vehicle
            line_strip.color.r = 0.0
            line_strip.color.g = 1.0  # Green for ego
            line_strip.color.b = 0.0
        elif vehicle_id == 1:  # Surround vehicle
            line_strip.color.r = 0.0
            line_strip.color.g = 0.0
            line_strip.color.b = 1.0  # Blue for surround
        line_strip.color.a = 0.8  # Higher opacity for better visibility

        # Populate the points in the line strip
        for point in trajectory.points:
            p = Point32()
            p.x = point.x
            p.y = point.y
            p.z = 0  # Flat on the ground
            line_strip.points.append(p)

        marker_array.markers.append(line_strip)

        # Circles (spheres) at each point for the vehicle
        point_id = vehicle_id * 1000  # Start ID for the points, unique per vehicle

        for point in trajectory.points:
            # Get the position of the point
            position = np.array([point.x, point.y])

            # Calculate circles based on vehicle_id and radius
            if vehicle_id == 0:  # Ego vehicle
                circles = self.calculate_circles(position, point.yaw, 1.86, 4.9, ego_radius)
            elif vehicle_id == 1:  # Surround vehicle
                circles = self.calculate_circles_surround(position, point.yaw, 0.8, 1.7, surround_radius)

            # For each circle, create a sphere marker
            for i, circle_pos in enumerate(circles):
                circle_marker = Marker()
                circle_marker.header.frame_id = "map"
                circle_marker.header.stamp = rospy.Time.now()
                circle_marker.ns = f"vehicle_{vehicle_id}_trajectory_points_fill"
                circle_marker.action = Marker.ADD
                circle_marker.pose.orientation.w = 1.0
                circle_marker.lifetime = rospy.Duration(1.0)
                circle_marker.id = point_id + i
                circle_marker.type = Marker.SPHERE  # Use SPHERE for the filled circle
                circle_marker.scale.x = 2*ego_radius if vehicle_id == 0 else 2*surround_radius  # Set radius dynamically
                circle_marker.scale.y = 2*ego_radius if vehicle_id == 0 else 2*surround_radius
                circle_marker.scale.z = 0.01  # Flat on the ground

                # Set the circle fill color based on vehicle_id (0 for ego, 1 for surround)
                circle_marker.color = line_strip.color  # Inherit color from line_strip

                # Set the position of the circle
                circle_marker.pose.position.x = circle_pos[0]
                circle_marker.pose.position.y = circle_pos[1]
                circle_marker.pose.position.z = 0

                marker_array.markers.append(circle_marker)

            point_id += 100  # Increment to ensure unique ID for the next point

        # Publish the MarkerArray to RViz
        self.trajectory_marker_pub.publish(marker_array)

    def save_ego_state(self, time, x, y):
        """Save only x, y coordinates of the ego vehicle to CSV."""
        try:
            with open(self.ego_state_file, 'a', newline='') as f:
                csv.writer(f).writerow([time, x, y])
        except Exception as e:
            rospy.logwarn(f"Failed to save ego state: {e}")

    def save_future_trajectory(self, time, x_values, y_values):
        """미래 경로를 time, x1, y1, x2, y2, ..., x30, y30 형식으로 저장합니다."""
        try:
            # x_values와 y_values를 교차로 병합하여 한 줄에 기록
            flattened_values = [val for pair in zip(x_values, y_values) for val in pair]
            row = [time] + flattened_values

            with open(self.future_trajectory_file, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(row)
            
            #rospy.loginfo(f"Saved future trajectory at time {time} with {len(x_values)} points")

        except Exception as e:
            rospy.logwarn(f"Failed to save future trajectory: {e}")

    def run(self):
        rate = rospy.Rate(20)  # 20 Hz
        time_horizon = 3.0  # Default time horizon
        dt = 0.2  # 0.2 seconds per step

        # vehicle dimension
        ego_vehicle_width = 1.86
        ego_vehicle_length = 4.9
        ego_vehicle_radius = 1.2 # 0.1 meter for each circle

        # ebt dimension  
        ebt_width = 0.8
        ebt_length = 1.7
        ebt_radius = 1.2

        while not rospy.is_shutdown():
            start_time = time.time()  # 루프 시작 시간 기록
            if self.ego_vehicle:
                #vehicle_id = 0  # ID counter for markers

                 # 고정된 vehicle_id 사용
                ego_vehicle_id = 0  # ID for ego vehicle
                surround_vehicle_id = 1  # ID for surround vehicle

                # Get the ego vehicle's yaw angle
                ego_yaw = self.ego_vehicle.yaw # rad/s

                # Predict Ego vehicle trajectory over the time horizon using the CV model
                ego_vehicle_copy_run = copy.deepcopy(self.ego_vehicle)
                #rospy.loginfo("Start sfdsfsdfsdf...")


                if len(self.data_buffer) == self.max_steps * 7:
                    input_tensor = torch.tensor(self.data_buffer, dtype=torch.float32).view(1, 30, 7).to(self.device)
                    ego_trajectory = self.predict_motion_LSTM(input_tensor,self.data_mean, self.data_std,dt)
                    #rospy.loginfo("Start predicting...")
                    # rospy.loginfo(ego_trajectory)
                    self.visualize_trajectory(ego_trajectory, ego_vehicle_id, ego_vehicle_radius, ebt_radius)  # ego_vehicle_id = 0
                
                #ego_trajectory = self.predict_motion_dynamics(self.ego_vehicle, self.ego_roll, time_horizon, dt)
                del ego_vehicle_copy_run
                
            elapsed_time = time.time() - start_time  # 루프 소요 시간 계산
            rospy.loginfo(f"Loop execution time: {elapsed_time:.6f} seconds")  # 소요 시간 출력

            rate.sleep()

if __name__ == "__main__":
    # rospy.init_node('collision_risk_node')
    collision_checker = CollisionChecker()  # CollisionChecker 클래스의 인스턴스를 생성

    try:
        collision_checker.run()  # 인스턴스의 run() 메서드 호출
    except rospy.ROSInterruptException:
        pass