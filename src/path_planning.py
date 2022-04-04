#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import time, os
from utils import LineTrajectory
from numpy import inf

class PathPlan(object):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic")
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        self.trajectory = LineTrajectory("/planned_trajectory")
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10)
        self.traj_pub = rospy.Publisher("/trajectory/current", PoseArray, queue_size=10)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)

        self.map_ready = False
        self.start_ready = False
        self.goal_ready = False

        self.map_width = None
        self.map_height = None
        self.map_resolution = None
        self.map_data = None

        self.start_point = None
        self.goal_point = None


    def map_cb(self, msg):
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_orientation = msg.info.origin.orientation
        self.map_position = msg.info.origin.position

        map_1d = np.array(msg.data)
        map_2d = map_1d.reshape((self.map_height, self.map_width))
        self.map_data = map_2d

        #TODO: Discretize the map to a lower resolution if necessary

        self.map_ready = True

    def odom_cb(self, msg):
        if self.map_ready:
            start_x = msg.pose.pose.position.x
            start_y = msg.pose.pose.position.y
            self.start_point = np.array([start_x, start_y])
            self.start_ready = True
            self.plan_path(self.start_point, self.goal_point, self.map_data)


    def goal_cb(self, msg):
        if self.map_ready:
            goal_x = msg.pose.position.x
            goal_y = msg.pose.position.y
            self.goal_point = np.array([goal_x, goal_y])
            self.goal_ready = True
            self.plan_path(self.start_point, self.goal_point, self.map_data)


    def plan_path(self, start_point, end_point, map):
        if self.map_ready and self.start_ready and self.goal_ready:
            #TODO: Convert the start and end point to pixels (or map to meters)
            #TODO: Run A*
            #TODO: Convert the results of A* to meters + transform to map frame

            # publish trajectory
            self.traj_pub.publish(self.trajectory.toPoseArray())

            # visualize trajectory Markers
            self.trajectory.publish_viz()


if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
