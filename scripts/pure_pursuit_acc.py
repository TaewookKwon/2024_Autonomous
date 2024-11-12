#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float64
from beginner_tutorials.msg import PIDGains
from nav_msgs.msg import Odometry
from morai_msgs.msg import CtrlCmd, EgoVehicleStatus


class pidControl:
    def __init__(self):
        # 기본값
        self.p_gain = 0.0
        self.i_gain = 0.0
        self.d_gain = 0.0
        self.prev_error = 0
        self.i_control = 0
        self.controlTime = 0.02

        # PID 게인 토픽 구독
        rospy.Subscriber("/pid_gains", PIDGains, self.update_gains)

    def update_gains(self, msg):
        # PID 게인 업데이트
        self.p_gain = msg.p_gain if msg.p_gain else 0.0
        self.i_gain = msg.i_gain if msg.i_gain else 0.0
        self.d_gain = msg.d_gain if msg.d_gain else 0.0

    def pid(self, target_acceleration, current_acceleration):
        error = target_acceleration - current_acceleration
        p_control = self.p_gain * error
        if error <= 5:
            self.i_control += self.i_gain * error * self.controlTime
        d_control = self.d_gain * (error - self.prev_error) / self.controlTime
        output = p_control + self.i_control + d_control
        self.prev_error = error
        return output


class acceleration_control:
    def __init__(self):
        rospy.init_node('acceleration_control', anonymous=True)

        self.ctrl_cmd_pub = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=1)
        rospy.Subscriber("odom", Odometry, self.odom_callback)
        rospy.Subscriber("Ego_topic", EgoVehicleStatus, self.status_callback)

        self.ctrl_cmd_msg = CtrlCmd()
        self.ctrl_cmd_msg.longlCmdType = 3
        self.target_acceleration = -5.0
        self.current_acceleration = 0.0

        self.pid_controller = pidControl()

        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.current_acceleration is not None:
                self.ctrl_cmd_msg.steering = 0.0
                output = self.pid_controller.pid(self.target_acceleration, self.current_acceleration)

                if output > 0:
                    self.ctrl_cmd_msg.accel = output
                    self.ctrl_cmd_msg.brake = 0
                else:
                    self.ctrl_cmd_msg.accel = 0
                    self.ctrl_cmd_msg.brake = -output
                    self.ctrl_cmd_msg.acceleration = self.target_acceleration

                self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)

            rate.sleep()

    def odom_callback(self, msg):
        self.current_acceleration = msg.twist.twist.linear.x

    def status_callback(self, msg):
        pass


if __name__ == '__main__':
    try:
        acceleration_control()
    except rospy.ROSInterruptException:
        pass
