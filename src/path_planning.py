#!/usr/bin/env python

import rospy
import numpy as np
import tf
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import time, os
from utils import LineTrajectory
from numpy import inf
import math

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

        self.box_size = 5 #Determines how granular to discretize the data

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
        #Extract all the info from the map message
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_orientation = msg.info.origin.orientation #Quaternion
        self.map_position = msg.info.origin.position #Point

        #Discretize the map
        self.map_data = self.discretize_map(self.map_height, self.map_width, np.array(msg.data))

        #Signal that the map has been loaded
        self.map_ready = True

    def discretize_map(self, height, width, data):
        #Replace all unknown grid spaces as fully occupied
        data[data == -1] = 100

        #Turn the data into a 2D grid
        map_2d = data.reshape((height, width))

        #Iterate through every nth row and nth column
        discretized_map_2d = np.zeros((height//self.box_size, width//self.box_size))
        for row in range(0, height-self.box_size+1, self.box_size):
            for col in range(0, width-self.box_size+1, self.box_size):
                #Take the average of each box_size by box_size square and make that the new value
                avg = np.average(map_2d[row:row + self.box_size, col:col + self.box_size])
                discretized_map_2d[row//self.box_size, col//self.box_size] = avg
        
        #Note: For Stata Basement, the rows are from bottom to top (index 0 = bottom of map) because the
        #orientation of the map's origin is rotated 180 degrees over the z-axis. This should resolve
        #itself when transforming the map frame (hopefully)
        return discretized_map_2d

    def odom_cb(self, msg):
        if self.map_ready:
            #Get the x and y position of the car from the odometry
            start_x = msg.pose.pose.position.x
            start_y = msg.pose.pose.position.y
            self.start_point = np.array([start_x, start_y])

            #Signal that the start position has been loaded
            self.start_ready = True

            #Attempt to plan a path
            self.plan_path(self.start_point, self.goal_point, self.map_data)


    def goal_cb(self, msg):
        if self.map_ready:
            #Get the x and y position of the goal from the 2D Nav Goal
            goal_x = msg.pose.position.x
            goal_y = msg.pose.position.y
            self.goal_point = np.array([goal_x, goal_y])

            #Signal that the goal position has been loaded
            self.goal_ready = True

            #Attempt to plan a path
            self.plan_path(self.start_point, self.goal_point, self.map_data)


    def plan_path(self, start_point, goal_point, map):
        if self.map_ready and self.start_ready and self.goal_ready:
            #Get the rotation matrix from the map frame to the image frame
            rot_mat = tf.transformations.quaternion_matrix([self.map_orientation.x, self.map_orientation.y, self.map_orientation.z, self.map_orientation.w])
            rot_mat = np.array([[rot_mat[0,0], rot_mat[0,1]], [rot_mat[1,0], rot_mat[1,1]]])

            #Transform the start and end points from the map frame to the image frame
            start_pixel = (np.dot(rot_mat, start_point) + np.array([self.map_position.x, self.map_position.y])) / self.map_resolution
            goal_pixel = (np.dot(rot_mat, goal_point) + np.array([self.map_position.x, self.map_position.y])) / self.map_resolution

            #Scale the pixels to their corresponding location in our discrete grid space (and flip x and y)
            discretized_start = (int(start_pixel[1]//self.box_size), int(start_pixel[0]//self.box_size))
            discretized_goal = (int(goal_pixel[1]//self.box_size), int(goal_pixel[0]//self.box_size))

            rospy.loginfo(discretized_start)

            #Run A* using the discretized start, goal, and map
            self.A_star(discretized_start, discretized_goal, map)

            #TODO: Run A*
            #TODO: Convert the results of A* to meters + transform to map frame

            # publish trajectory
            self.traj_pub.publish(self.trajectory.toPoseArray())

            # visualize trajectory Markers
            self.trajectory.publish_viz()

    def A_star(self, start, goal, map):
        heuristic = lambda start, goal: math.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2)
        open = {start}
        segments = {}
        cost = {start: 0}
        score = {start: heuristic(start, goal)}

        while open:
            curr = min(score, key=score.get)
            if curr == goal:
                return self.get_path(segments, curr)

    def get_path(self, segments, node):
        pass


if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
