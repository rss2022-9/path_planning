#!/usr/bin/env python2

from turtle import right
import numpy as np
import message_filters
import rospy
import math
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from visualization_tools import *

class Safety:
    # topic parameters and variables
    scan_topic = rospy.get_param("~scan_topic", "/scan")
    drive_topic = rospy.get_param("~drive_topic", "/drive")
    safety_topic = rospy.get_param("~safety_topic", "/vesc/low_level/ackermann_cmd_mux/input/safety")

    prev_velocity = 0

    def __init__(self):
        # Dual subscription only applies once we have the actual vehicle
        self.lsr_sub = message_filters.Subscriber(self.scan_topic, LaserScan) # subscribes to scan
        self.drv_sub = message_filters.Subscriber(self.drive_topic, AckermannDriveStamped) # subscribes to the current state of the car
        ts = message_filters.ApproximateTimeSynchronizer([self.lsr_sub, self.drv_sub], 5, 0.1) # syncs the two sub topics
        ts.registerCallback(self.safety_controller) # calls the callback function with two inputs

        self.drv_pub = rospy.Publisher(self.safety_topic, AckermannDriveStamped, queue_size=10)
        self.test_pub = rospy.Publisher("debug1", String, queue_size=10)

        # visual publisher example
        self.left_lane_pub = rospy.Publisher("/left_lane", Marker, queue_size=1)
        self.right_lane_pub = rospy.Publisher("/right_lane", Marker, queue_size=1)

    def safety_controller(self, scan, drive_stamp):
        steering_angle = drive_stamp.drive.steering_angle
        steering_angle_v = drive_stamp.drive.steering_angle_velocity
        speed = drive_stamp.drive.speed
        acceleration = drive_stamp.drive.acceleration
        jerk = drive_stamp.drive.jerk

        # Calculate turn radius
        L = 0.25 #Length from center of wheelbase to front of car
        R = L / max(0.005, math.tan(steering_angle), key=abs)

        # Obtain relevant LIDAR data
        ranges, intensities = np.array(scan.ranges), np.array(scan.intensities)
        angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)
        pol_coords = np.stack([ranges, angles, intensities], axis=1)
        r, theta, i = pol_coords.T

        # Calculate x and y values for each data point from the LIDAR scan within a certain range
        x = r*np.cos(theta)
        x_less = np.logical_and(abs(x) < (abs(R) - 0.25), x >= 0)
        x = x[x_less]
        y = r*np.sin(theta)
        y = y[x_less]
        r = r[x_less]

        # Predict travel lanes from the left and right side of the car
        sign = 1 if steering_angle < 0 else -1
        right_lane_y = sign * np.sqrt((R + 0.25) ** 2 - np.square(x)) + R
        left_lane_y = sign * np.sqrt((R - 0.25) ** 2 - np.square(x)) + R

        # If a LIDAR point falls between the two lanes within a certain distance, trigger safety controls
        stop_distance = rospy.get_param("~stop_dist_factor", 0.1) * self.prev_velocity**2 + rospy.get_param("~stop_dist", 0.5)
        trigger_safety_controls = False
        y_safety_cond = np.logical_and(y < left_lane_y, y > right_lane_y)
        r_safety_cond = r < stop_distance

        # trigger safety control
        trigger_safety_controls = np.logical_and(r_safety_cond, y_safety_cond).any()

        #Stop if safety controls triggerred, then gradually speed back up
        if trigger_safety_controls:
            rospy.loginfo("STOP!")
            new_velocity = 0 #max(0, self.prev_velocity / 2 - 0.1)
            self.prev_velocity = new_velocity
            self.drv_pub.publish(drive_cmd_maker(new_velocity, acceleration, jerk, steering_angle, steering_angle_v))
        elif self.prev_velocity != speed:
            new_velocity = min(self.prev_velocity + 0.1, speed)
            self.prev_velocity = new_velocity
            self.drv_pub.publish(drive_cmd_maker(new_velocity, acceleration, jerk, steering_angle, steering_angle_v))

        # Visualization of left and right lanes
        viz_x = np.array([(abs(R) - 0.25) * i / 10.0 for i in range(10)] if abs(R) < 5 else [i / 2.0 for i in range(6)])
        viz_y1 = np.array(sign * np.sqrt((R + 0.25)**2 - np.square(viz_x)) + R)
        viz_y2 = np.array(sign * np.sqrt((R - 0.25)**2 - np.square(viz_x)) + R)
        VisualizationTools.plot_line(viz_x, viz_y1, self.left_lane_pub, frame="/base_link")
        VisualizationTools.plot_line(viz_x, viz_y2, self.right_lane_pub, frame="/base_link")

def drive_cmd_maker(speed=0.0, acceleration=0.0, jerk=0.0, steering_angle=0.0, steering_angle_v=0.0):
    """
    :param steering_angle: desired virtual angle (radians)
    :param steering_angle_v: desired rate of change (radians/s), 0 is max
    :param speed: desired forward speed (m/s)
    :param acceleration: desired acceleration (m/s^2)
    :param jerk: desired jerk (m/s^3)

    :return: AckermannDriveStamped type to publish
    """
    drv_cmd = AckermannDriveStamped()
    drv_cmd.header.stamp = rospy.get_rostime()
    drv_cmd.header.frame_id = "base_link"

    drv_cmd.drive.steering_angle = float(steering_angle)
    drv_cmd.drive.steering_angle_velocity = float(steering_angle_v)
    drv_cmd.drive.speed = speed
    drv_cmd.drive.acceleration = acceleration
    drv_cmd.drive.jerk = jerk

    return drv_cmd

if __name__ == "__main__":
    rospy.init_node('safety')
    safety = Safety()
    rospy.spin()

