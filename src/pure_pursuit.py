#!/usr/bin/env python

import rospy
import numpy as np
import time
import utils
import tf
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from tf.transformations import euler_from_quaternion

from geometry_msgs.msg import PoseArray, PoseStamped
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_tools import *
from nav_msgs.msg import Odometry

class PurePursuit(object):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic")
        self.trajectory = utils.LineTrajectory("/followed_trajectory")
        point_topic = "/pp/point"
        self.line_pub = rospy.Publisher(point_topic, Marker, queue_size=1)
        self.drive_pub = rospy.Publisher("/drive", AckermannDriveStamped, queue_size=1)
        
        self.traj_sub = rospy.Subscriber("/trajectory/current", PoseArray, self.trajectory_callback, queue_size=1)
        self.move_car = rospy.Subscriber("/pf/pose/odom", Odometry, self.PPController, queue_size=1)

        self.lookahead        = 1.0
        self.speed            = 1
        self.wheelbase_length = 0.325
        self.DIST_THRESH = 0.01
        self.path_points_set = False
        self.car_pose = None
        self.path_points  = None
        self.car_ang = None
        self.path_line = None
        
    def trajectory_callback(self, msg):
        ''' Clears the currently followed trajectory, and loads the new one from the message
        '''
        print "Receiving new trajectory:", len(msg.poses), "points"
        self.trajectory.clear()
        path_points_low = np.array(self.fromPoseArray(msg))
        x = path_points_low[:,0]
        y = path_points_low[:,1]
        x_high, y_high = self.populateLine(x,y)
        self.path_points = np.array([x_high,y_high])
        points = self.path_points.shape[0]-1
        coeff = np.polyfit(x_high,x_high,points)
        self.path_line = np.poly1d(coeff)
        self.path_points_set = self.path_points is not None
        self.trajectory.publish_viz(duration=3000.0)
        return
        
   
    

    def find_x(self, car_pose, car_ang):
        # Closest point of every segment
        def circe(x):
            rad = self.lookahead
            x_offset = -car_pose[1]
            y_offset = car_pose[0]
            return (np.sqrt(-(x-x_offset)**2 + (rad**2)) - y_offset)

        
        path_points = np.transpose(self.path_points)
        points_rel = path_points - car_pose
        distances = np.flip(np.linalg.norm(points_rel, axis=1),0)
        flipindex = np.argmax(distances <= self.lookahead)
        index = (distances.shape[0] - 1) - flipindex
        #output = fsolve(lambda x:self.path_line(x)-circe(x),car_pose[0])
        target_mark = path_points[index,:]
        target = points_rel[index,:]
        mag = distances[flipindex]
        x, y = target[0], target[1]
        ang =  np.arctan2(y,x)
        rel_ang = car_ang-ang
        rel_x =  x*np.cos(rel_ang) - y*np.sin(rel_ang)
        VisualizationTools.plot_point(target_mark[0], target_mark[1], self.line_pub, frame="/map")
        print(rel_ang*180/np.pi,car_ang*180/np.pi,ang*180/np.pi)
        #print(rel_x)
        return rel_x,mag
        


    # Pure pursuit controller
    def PPController(self, car_odom):
        if self.path_points_set:
            car_pose = [car_odom.pose.pose.position.x, car_odom.pose.pose.position.y]
            car_odom_ang = [car_odom.pose.pose.orientation.x,car_odom.pose.pose.orientation.y,car_odom.pose.pose.orientation.z,car_odom.pose.pose.orientation.w]
            roll,pitch,car_ang = euler_from_quaternion(car_odom_ang)
            x,dist= self.find_x(car_pose, car_ang)
            
            tr = (dist)/(2*x) if x != 0 else 0
            ang = -np.arctan(self.wheelbase_length/tr)
            output = max(min(ang,0.34),-0.34)
            drive_cmd = AckermannDriveStamped()
            drive_cmd.drive.speed = 0
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

        alpha = np.linspace(0, 1, 10000)
        x_regular, y_regular = fx(alpha), fy(alpha)
        return x_regular, y_regular


if __name__=="__main__":
    rospy.init_node("pure_pursuit")
    pf = PurePursuit()
    rospy.spin()
