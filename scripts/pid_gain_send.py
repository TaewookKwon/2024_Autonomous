#!/usr/bin/env python3
import rospy
from beginner_tutorials.msg import PIDGains

def send_pid_gains(p, i, d):
    rospy.init_node('pid_gains_publisher', anonymous=True)
    pub = rospy.Publisher('/pid_gains', PIDGains, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz

    while not rospy.is_shutdown():
        pid_msg = PIDGains()
        pid_msg.p_gain = p
        pid_msg.i_gain = i
        pid_msg.d_gain = d
        pub.publish(pid_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        send_pid_gains(0.00, 0.00, 0.00)
    except rospy.ROSInterruptException:
        pass