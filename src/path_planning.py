#!/usr/bin/env python

import rospy
import numpy as np
import tf
from geometry_msgs.msg import PoseStamped, PoseArray, Point
from visualization_msgs.msg import Marker
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

        self.algorithm = "A_star" # which search algorithm to use "A_star" and "RRT"

        self.box_size = 4 # Determines how granular to discretize the data, A* default = 10
        self.occupied_threshold = 3 #Probability threshold to call a grid space occupied (0 to 100)
        self.padding_size = 4 #Amount of padding to add to walls when path planning (Max Value = 6 for Stata Basement)

        # parameters for RRT
        self.max_distance = 20   # max distance from new node to old node, unit in pixel
        self.car_to_pixel = 1     # car width in pixel unit, used for collision detection
        self.target_range = 5     # close in L1 norm of the target, then finish RRT

        self.map_ready = False
        self.start_ready = False
        self.goal_ready = False

        self.map_width = None
        self.map_height = None
        self.map_resolution = None
        self.map_data = None
        self.dmap_width = None # width of discretized map
        self.dmap_height= None # height of discretized map

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
        self.dmap_height, self.dmap_width = self.map_data.shape

        #Signal that the map has been loaded
        self.map_ready = True
        rospy.loginfo("Map Loaded")

    def discretize_map(self, height, width, data):
        #Turn the data into a 2D grid
        map_2d = data.reshape((height, width))
        
        # #Add a boundary to the map walls
        map_2d_copy = map_2d.copy()
        for row in range(self.padding_size, self.map_height - self.padding_size):
            for col in range(self.padding_size, self.map_width - self.padding_size):
                if map_2d_copy[row, col] > self.occupied_threshold:
                    for i in range(-self.padding_size, self.padding_size + 1):
                        for j in range(-self.padding_size, self.padding_size + 1):
                            map_2d[row + i, col + j] = 100
        
        #Replace all unknown grid spaces as fully occupied
        map_2d[map_2d == -1] = 100

        #Iterate through every nth row and nth column
        discretized_map_2d = np.zeros((height//self.box_size, width//self.box_size))
        for row in range(0, height-self.box_size+1, self.box_size):
            for col in range(0, width-self.box_size+1, self.box_size):
                #Take the average of each box_size by box_size square and make that the new value
                avg = np.average(map_2d[row:row + self.box_size, col:col + self.box_size])
                discretized_map_2d[row//self.box_size, col//self.box_size] = avg
        
        #Note: For Stata Basement, the rows are from bottom to top (index 0 = bottom of map) because the
        #orientation of the map's origin is rotated 180 degrees over the z-axis. This should resolve
        #itself when transforming the map frame
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
            rospy.loginfo("Goal point set")

            #Attempt to plan a path
            self.plan_path(self.start_point, self.goal_point, self.map_data)


    def plan_path(self, start_point, goal_point, map):
        if self.map_ready and self.start_ready and self.goal_ready:
            #Convert the start and goal point into their discretized coordinates
            discretized_start = self.xy_to_discretized(start_point)
            discretized_goal = self.xy_to_discretized(goal_point)

            #Run search algorithm using the discretized start and goal
            rospy.loginfo(self.algorithm + " is starting planning")
            if self.algorithm is "RRT":
                uv_path = self.RRT_search(discretized_start, discretized_goal)
            else:
                uv_path = self.A_star(discretized_start, discretized_goal)

            if uv_path is not None:
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

                #Wait for a new goal position to be set
                self.goal_ready = False
            else:
                rospy.loginfo("No path found")

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
        cost_func = lambda node1, node2: heuristic_func(node1, node2) if self.map_data[node2[0],node2[1]] < self.occupied_threshold else np.inf
        
        open = {start} #Set of nodes discovered so far
        segments = {} #Map of nodes to the prior node they were found by (used for path reconstruction)
        cost = {start: 0} #Cost of the path to each node so far
        score = {start: heuristic_func(start, goal)} #Cost of the path to each node + heuristic for remaining distance

        while open:
            #Get the min score node
            curr = min(open, key=score.get)

            #If the node is the goal node, reconstruct the path
            if curr == goal:
                return self.get_A_star_path(segments, curr)

            #Explore the neighbors of the current node
            open.remove(curr)
            for node in self.get_neighbors(curr):
                node = tuple(node)
                #If the path to this node is min cost so far, record it
                temp_cost = cost.get(curr, np.inf) + cost_func(curr, node)
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

    def get_A_star_path(self, segments, node):
        path = [node]
        while node in segments.keys():
            node = segments[node]
            path.append(node)
        path.reverse()
        return path

    def RRT_search(self, start, goal):

        tree_pos = []      # position of each leaf 
        tree_parent = []   # parent of each leaf
        tree = {"pos":tree_pos, "parent":tree_parent}
        start = np.array(start)
        goal = np.array(goal)
        tree_pos.append(start)
        tree_parent.append(-1)
        
        while True:
            x_rand  = self.rand_sample()                    # randomly sample new point
            nearest = self.find_nearest_leaf(x_rand, tree_pos)  # find the nearest leaf(index) to sampled point
            if nearest == -1:
                continue
            leaf_new = self.steer(x_rand, tree_pos[nearest], self.max_distance)      # setup new leaf pos, we don't want to go too far and have the obstacle. Also possible to include dynamic?

            if self.obstacle_free(tree_pos[nearest], leaf_new):
                tree_pos.append(leaf_new)
                tree_parent.append(nearest)
                diff = leaf_new-goal
                self.trajectory.publish_RRT_edge(leaf_new, tree_pos[nearest])
                if diff.dot(diff) < self.target_range**2:
                    return self.get_RRT_path(tree)
 
    def rand_sample(self):
        # randomly sample point in the map
        # TODO: setup voronoi bias in the future
        while True:
            col = np.random.random_sample()*(self.dmap_width-1)  # col
            row = np.random.random_sample()*(self.dmap_height-1) # row
            col = int(np.round(col))
            row = int(np.round(row))
            if self.map_data[row, col] >= 0 and self.map_data[row,col] < self.occupied_threshold:  # grid is defined and unoccupied
                return np.array([row, col])

    def find_nearest_leaf(self, x_rand, tree_pos):

        diff = x_rand - np.array(tree_pos)
        square = diff*diff  # element multiplication
        square_distance = np.sum(square, axis=1)
        min_val = np.min(square_distance)
        if min_val > 4:                          # avoid sample point overlaps, discretized map, at least two step forward
            min_ind = np.where(square_distance==min_val)
            # rospy.loginfo(min_ind)
            return min_ind[0][-1]                # prefer to newest point
        else:
            return -1
        # nearest = -1
        # dist    = 0
        # min_dist= np.inf

        # for leaf in tree_pos:
        #     rospy.loginfo(leaf)
        #     dist = leaf - x_rand  # delta
        #     dist = dist.dot(dist)    # distance square
        #     if dist<min_dist and dist>4:  # avoid sample point overlaps, discretized map, at least two step forward
        #         min_dist = dist
        #         nearest = tree_pos.index(leaf)

        # # assert nearest > 0, "fail to find nearest leaf"
        # rospy.loginfo(nearest)
        # return nearest

    def steer(self, x_rand, leaf_nearest, delta):
        # x_rand and leaf_nearest decide direction of vector, delta is the norm
        # TODO: include dynamics, making traj smoother
        diff = x_rand - leaf_nearest
        if diff.dot(diff) < self.max_distance**2:
            return x_rand
        direction = diff / np.sqrt(diff.dot(diff))  # normalization
        x_new = np.round(leaf_nearest + delta*direction)
        return x_new.astype(int)

    def obstacle_free(self, pointA, pointB):
        """
        check if there is an obstacle between A and B"""
        # TODO: current version is very conservative using box of A-B as constraint (similar to AABB algorithm)
        x_min = pointA[0] if pointA[0] < pointB[0] else pointB[0]
        x_max = pointA[0] if pointA[0] > pointB[0] else pointB[0]
        y_min = pointA[1] if pointA[1] < pointB[1] else pointB[1]
        y_max = pointA[1] if pointA[1] > pointB[1] else pointB[1]
        x_min = int(np.round(x_min))
        x_max = int(np.round(x_max))+1
        y_min = int(np.round(y_min))
        y_max = int(np.round(y_max))+1
        rectangle = self.map_data[x_min:x_max, y_min:y_max]
        # rospy.loginfo(rectangle.shape)
        if np.max(rectangle) > self.occupied_threshold or np.min(rectangle) < 0:  # ocupied or undefined
            return False
        else:
            return True

    def get_RRT_path(self, tree):
        path = []
        pos = tree["pos"]
        parent = tree["parent"]
        index = len(parent) - 1  # start from the goal

        while index != -1:          # end when meeting first node
            path.append(pos[index])
            index = parent[index]   # parent index
        path.reverse()
        return path

if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
