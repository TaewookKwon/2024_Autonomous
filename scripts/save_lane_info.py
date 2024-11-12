#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from beginner_tutorials.msg import TrackingPoint
import time
import csv
from scipy.spatial import distance

class LaneInfoProcessor:
    def __init__(self):
        # 텍스트 파일 경로
        self.path = '/home/taewook/catkin_ws/src/beginner_tutorials/path/'
        self.lane_files = ['lane1.txt', 'lane2.txt', 'lane3.txt']

        # 각각의 lane 데이터를 저장할 리스트
        self.lane_data = [self.load_lane_data(file) for file in self.lane_files]

        # CSV 파일을 생성하고 헤더를 작성합니다.
        self.file_name = '/home/taewook/catkin_ws/src/beginner_tutorials/data/lane_avg_data.csv'
        with open(self.file_name, 'w', newline='') as csvfile:
            self.csvwriter = csv.writer(csvfile)
            self.csvwriter.writerow(['time', 'lane1_avg', 'lane2_avg', 'lane3_avg'])

        # 시간 관리 변수
        self.last_logged_time = None
        self.next_log_time = None  # 다음 로그를 저장할 정확한 시간을 관리

        # ROS 구독자 설정
        self.subscriber = rospy.Subscriber('/ego_tracking', TrackingPoint, self.callback)

        # 저장할 결과
        self.lane_averages = [None, None, None]

    def load_lane_data(self, file_name):
        # 텍스트 파일에서 lane 데이터를 읽어옴
        lane_points = []
        with open(self.path + file_name, 'r') as file:
            reader = csv.reader(file, delimiter=' ')
            for row in reader:
                x, y, z = map(float, row)
                lane_points.append([x, y, z])
        return np.array(lane_points)  # NumPy 배열로 변환하여 반환

    def find_closest_points(self, lane, ego_x, ego_y, num_points=50):
        # 에고 차량 위치와 lane 사이의 거리를 계산하여 가장 가까운 점 num_points개를 찾음
        distances = distance.cdist(lane[:, :2], [[ego_x, ego_y]], 'euclidean').flatten()
        closest_indices = np.argsort(distances)[:num_points]
        closest_points = lane[closest_indices]
        return closest_points

    def callback(self, msg):
        start_time = time.time()  # 시작 시간 기록

        # 에고 차량의 위치 정보
        ego_x = msg.x
        ego_y = msg.y

        # ROS 메시지에서 시간 정보를 float으로 변환 (초 단위)
        current_time = msg.time.secs + msg.time.nsecs * 1e-9

        # 다음 로그 저장 시간을 정밀하게 계산
        if self.next_log_time is None:
            self.next_log_time = current_time + 0.1

        # 각각의 lane에 대해 가장 가까운 50개의 점을 찾고 평균을 계산
        for i, lane in enumerate(self.lane_data):
            closest_points = self.find_closest_points(lane, ego_x, ego_y, num_points=50)
            avg_point = np.mean(closest_points, axis=0)
            self.lane_averages[i] = avg_point  # 각 레인의 평균 좌표를 저장

        # 정확하게 0.1초 단위로만 저장 (next_log_time과 비교하여)
        if current_time >= self.next_log_time:
            with open(self.file_name, 'a', newline='') as csvfile:
                self.csvwriter = csv.writer(csvfile)
                self.csvwriter.writerow([
                    current_time, 
                    self.lane_averages[0][0],  # lane1 평균의 x 좌표
                    self.lane_averages[1][0],  # lane2 평균의 x 좌표
                    self.lane_averages[2][0]   # lane3 평균의 x 좌표
                ])
            self.next_log_time += 0.1  # 다음 로그 저장 시간 갱신

        # 디버깅을 위한 실행 시간 출력
        end_time = time.time()
        rospy.loginfo(f"Calculation completed in {end_time - start_time:.4f} seconds")

        # 터미널에 계산한 3개의 점 출력
        for i, avg in enumerate(self.lane_averages):
            rospy.loginfo(f"Lane {i + 1} average point: x={avg[0]:.3f}, y={avg[1]:.3f}, z={avg[2]:.3f}")

if __name__ == '__main__':
    rospy.init_node('lane_info_processor', anonymous=True)
    processor = LaneInfoProcessor()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down lane info processor node")
