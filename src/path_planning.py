#!/usr/bin/env python

import rospy
import numpy as np
import tf
from geometry_msgs.msg import PoseStamped, PoseArray, Point
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

        self.box_size = 10 #Determines how granular to discretize the data
        self.occupied_threshold = 3 #Probability threshold to call a grid space occupied (0 to 100)

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
            #Convert the start and goal point into their discretized coordinates
            discretized_start = self.xy_to_discretized(start_point)
            discretized_goal = self.xy_to_discretized(goal_point)

            #Run A* using the discretized start and goal
            uv_path = self.A_star(discretized_start, discretized_goal)

            #Convert path from (u,v) pixels to (x,y) coordinates in the map frame
            xy_path = []
            for coord in uv_path:
                xy_coord = self.discretized_to_xy(coord)
                xy_path.append(xy_coord)

                point = Point()
                point.x = xy_coord[0]
                point.y = xy_coord[1]
                self.trajectory.addPoint(point)

            # publish trajectory
            self.traj_pub.publish(self.trajectory.toPoseArray())

            # visualize trajectory Markers
            self.trajectory.publish_viz()

    def xy_to_discretized(self, coord):
        #Get the rotation matrix from the map frame to the image frame
        rot_mat = tf.transformations.quaternion_matrix([self.map_orientation.x, self.map_orientation.y, self.map_orientation.z, self.map_orientation.w])
        rot_mat = np.array([[rot_mat[0,0], rot_mat[0,1]], [rot_mat[1,0], rot_mat[1,1]]])

        #Transform the coordinate from the map frame to the image frame
        pixel = (np.dot(rot_mat, coord) + np.array([self.map_position.x, self.map_position.y])) / self.map_resolution

        #Scale the pixel to its corresponding location in our discrete grid space (and flip x and y)
        discretized = (int(pixel[1]//self.box_size), int(pixel[0]//self.box_size))

        return discretized

    def discretized_to_xy(self, coord):
        #Get the rotation matrix from the image frame to the map frame
        quat_inverse = tf.transformations.quaternion_inverse([self.map_orientation.x, self.map_orientation.y, self.map_orientation.z, self.map_orientation.w])
        rot_mat = tf.transformations.quaternion_matrix(quat_inverse)
        rot_mat = np.array([[rot_mat[0,0], rot_mat[0,1]], [rot_mat[1,0], rot_mat[1,1]]])

        #Get the xy value of the given coordinate
        xy_untranslated = np.array([coord[1], coord[0]]) * self.box_size * self.map_resolution

        #Translate and rotate the xy coordinate into the map frame
        xy_translated = xy_untranslated - np.array([self.map_position.x, self.map_position.y])
        xy = np.dot(rot_mat, xy_translated)

        return tuple(xy)

    def A_star(self, start, goal):
        #Heuristic function based on the straight line distance from start to goal
        heuristic_func = lambda start, goal: math.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2)
        #Cost function of 1 to move to an unoccupied cell, infinity to moved to an occupied one
        cost_func = lambda node: 1 if self.map_data[node[0],node[1]] < self.occupied_threshold else np.inf
        
        open = {start} #Set of nodes discovered so far
        segments = {} #Map of nodes to the prior node they were found by (used for path reconstruction)
        cost = {start: 0} #Cost of the path to each node so far
        score = {start: heuristic_func(start, goal)} #Cost of the path to each node + heuristic for remaining distance

        while open:
            #Get the min score node
            curr = min(open, key=score.get)

            #If the node is the goal node, reconstruct the path
            if curr == goal:
                return self.get_path(segments, curr)

            #Explore the neighbors of the current node
            open.remove(curr)
            for node in self.get_neighbors(curr):
                node = tuple(node)
                #If the path to this node is min cost so far, record it
                temp_cost = cost.get(curr, np.inf) + cost_func(node)
                if temp_cost < cost.get(node, np.inf):
                    segments[node] = curr
                    cost[node] = temp_cost
                    score[node] = temp_cost + heuristic_func(node, goal)
                    if node not in open:
                        open.add(node)
    
    def get_neighbors(self, node):
        #Add the 8 adjacent cells (vertical, horizontal, and diagonal) to the set of neighbors
        neighbors = np.array([
            (node[0]+1, node[1]+1),
            (node[0]+1, node[1]),
            (node[0]+1, node[1]-1),
            (node[0], node[1]+1),
            (node[0], node[1]-1),
            (node[0]-1, node[1]+1),
            (node[0]-1, node[1]),
            (node[0]-1, node[1]-1)])
        
        #Filter any neighbors that are out of the map
        neighbors = neighbors[neighbors[:,0] >= 0]
        neighbors = neighbors[neighbors[:,0] < self.map_height]
        neighbors = neighbors[neighbors[:,1] >= 0]
        neighbors = neighbors[neighbors[:,1] < self.map_width]

        return neighbors


    def get_path(self, segments, node):
        path = [node]
        while node in segments.keys():
            node = segments[node]
            path.append(node)
        path.reverse()
        return path

if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
