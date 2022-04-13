#!/usr/bin/env python

import rospy
import numpy as np
import time
import utils
import tf

from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseArray, PoseStamped
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from scipy.interpolate import interp1d
from visualization_tools import *
from nav_msgs.msg import Odometry

class PurePursuit(object):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic")
        self.trajectory = utils.LineTrajectory("/followed_trajectory")
        point_topic = "/pp/point"
        line_topic = "/pp/line"
        self.point_pub = rospy.Publisher(point_topic, Marker, queue_size=10)
        self.line_pub = rospy.Publisher(line_topic, Marker, queue_size=10)
        self.drive_pub = rospy.Publisher("/vesc/high_level/ackermann_cmd_mux/input/nav_0", AckermannDriveStamped, queue_size=10)
        self.lookahead        = 1.0
        self.speed            = 1.0
        self.wheelbase_length = 0.325
        self.DIST_THRESH = 0.01
        self.num_samples_add = 200
        self.path_points_set = False
        self.path_points  = None
        self.next_mark = None
        self.cur_point_index = None
        self.point1 = None
        self.point2 = None
        self.traj_sub = rospy.Subscriber("/trajectory/current", PoseArray, self.trajectory_callback, queue_size=10)
        self.move_car = rospy.Subscriber("/pf/pose/odom", Odometry, self.PPController, queue_size=10)
        
    def trajectory_callback(self, msg):
        ''' Clears the currently followed trajectory, and loads the new one from the message
        '''
        print "Receiving new trajectory:", len(msg.poses), "points"
        self.trajectory.clear()
        # This is an upsample of the path to smooth out the cars movements at certain points in the map
        path_points = np.array(self.fromPoseArray(msg))
        x = path_points[:,0]
        y = path_points[:,1]
        x_up,y_up = self.populateLine(x,y) # stack overflow code to upsample I don't understand it and you don't have to
        self.path_points = np.transpose(np.array([x_up,y_up])) # Upsampled values are the new path points
        self.path_points_set = self.path_points is not None
        self.next_mark = None
        self.cur_point_index = None
        self.point1 = None
        self.point2 = None
        self.started = False
        self.trajectory.publish_viz(duration=1.0)
        return
        
    def find_target(self, car_pose, car_ang):
        # Calculate closest point of every line segment
        # math here -> http://paulbourke.net/geometry/pointlineplane/
        # code for single segment here -> https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment/1501725#1501725
        # below is the same concept but vectorized
        path_points = self.path_points
        last_but_index = path_points.shape[0]-2
        last_index = path_points.shape[0]-1
        final_point = path_points[last_index,:]
        
        P1 = path_points[:-1,:]
        P2 = path_points[1:,:]
        P3 = car_pose
        LVEC = P2 - P1
        norm = np.linalg.norm(LVEC,axis=1) 
        t = np.einsum('ij,ij->i',LVEC,P3-P1)/(norm**2) # np.dot doesn't have an axis input so you have to do this wonky thing
        t = np.clip(t,0,1)
        min_vecs = P1 + t[:,np.newaxis]*LVEC
        min_vecs = min_vecs - P3
        distances = np.linalg.norm(min_vecs,axis=1)
        start_ind = np.argmin(distances)
        final_ind = start_ind + 1
        self.cur_point_index = start_ind
         
        if  (start_ind!=last_but_index): # don't bother if we're on the last segment
            point1 = path_points[start_ind,:]
            point2 = path_points[final_ind,:]
            self.next_mark = final_ind
            self.started = True
            
            # After finding closest segment check the furthers point within lookahead range
            # if you find points make the last one the actual start of the path so you start curving to the right trajectory
            next_mark = self.next_mark
            nextpoint = path_points[next_mark,:]
            mag = np.linalg.norm(nextpoint-car_pose)
            if mag < self.lookahead:
                next_mark = np.argmin(mag<self.lookahead) + next_mark
                point1 = path_points[next_mark,:]
                point2 = path_points[next_mark+1,:]
            self.point1 = point1
            self.point2 = point2
            

        # Stopping condition check is the end of the current path the last point?
        # Are you close enough to it?
        mag_final = np.linalg.norm(final_point-car_pose)
        if (self.cur_point_index >= last_but_index) and (mag_final <= 0.5):
            speed_multi = 0.0
        else:
            speed_multi = 1.0

        # After fiding the right line segmennt you have start and end of line segment
        # find the interesection of the line and lookahead circle
        # code here -> https://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm/86428#86428
        # below modified for arrays
        P1 = self.point1
        P2 = self.point2
        Q = car_pose
        r = self.lookahead
        V = P2 - P1
        
        a = np.dot(V,V)
        b = 2*np.dot(V,P1-Q)
        c = np.dot(P1,P1) + np.dot(Q,Q) - 2*np.dot(P1,Q) - r**2
        disc = b**2 - 4 * a * c
        if disc < 0:
            return None
        sqrt_disc = np.sqrt(disc)
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)
        t = max(t1,t2) # Larger value along line is the answer given the math should remain between 0 and 1 since we made sure every look ahead intersect is on a line
        target = P1 + t*V # Gives the actual location of the intersection of interest in the world frame

        # Calculate relative target between car and target location
        # Math described here -> https://imgur.com/a/Z6lwoM7
        rel_target = target - car_pose
        
        x, y = rel_target[0], rel_target[1]
        ang =  np.arctan2(y,x)
        mag = np.linalg.norm(rel_target)
        rel_ang = car_ang - ang
        
        rel_x =  mag*np.sin(rel_ang)
        
        VisualizationTools.plot_point(target[0], target[1], self.point_pub, frame="/map") # Visualize target point
        return rel_x, speed_multi

    # Pure pursuit controller
    def PPController(self, car_odom):
        if self.path_points_set:
            car_tran = [car_odom.pose.pose.position.x, car_odom.pose.pose.position.y]
            car_quat = [car_odom.pose.pose.orientation.x, car_odom.pose.pose.orientation.y, car_odom.pose.pose.orientation.z, car_odom.pose.pose.orientation.w]
            (roll, pitch, yaw) = euler_from_quaternion(car_quat)
            VisualizationTools.plot_line(self.path_points[:,0], self.path_points[:,1], self.line_pub, frame="/map") # Visualize path
            #print(car_tran)
            #exit()
            x, speed_multi= self.find_target(car_tran, yaw)
            
            tr = (self.lookahead**2)/(2*x) if x != 0 else 0 # Turning radius from relative x
            ang = -np.arctan(self.wheelbase_length/tr) # Angle from turning radius
            speed = self.speed*speed_multi
            output = max(min(ang,0.34),-0.34)
            drive_cmd = AckermannDriveStamped()
            drive_cmd.header.stamp = rospy.Time.now()
            drive_cmd.header.frame_id = 'base_link'
            drive_cmd.drive.speed = speed
            print(speed,ang)
            drive_cmd.drive.steering_angle = output
            self.drive_pub.publish(drive_cmd)

        return

    def fromPoseArray(self, trajMsg):
        points = []
        for p in trajMsg.poses:
            points.append((p.position.x, p.position.y))
        return points

    def populateLine(self,x,y):
        distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
        distance = distance/distance[-1]
        fx, fy = interp1d( distance, x ), interp1d( distance, y )
        num_points = len(x) + abs(self.num_samples_add)
        alpha = np.linspace(0, 1, num_points)
        x_regular, y_regular = fx(alpha), fy(alpha)
        return x_regular, y_regular
        
if __name__=="__main__":
    rospy.init_node("pure_pursuit")
    pf = PurePursuit()
    rospy.spin()
