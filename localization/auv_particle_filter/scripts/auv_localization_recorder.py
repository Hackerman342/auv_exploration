#!/usr/bin/env python3

import os

import rospy
import time
from geometry_msgs.msg import Pose, PoseArray, PoseWithCovarianceStamped
from geometry_msgs.msg import Transform, Quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Header

import keyboard  # using module keyboard


class AUV_Recorder(object):

    def __init__(self):

        root_folder = '/home/robot/smarc/auv_pf_results/'

        gt_odom_topic = '/gt/odom' # Odometry
        sim_odom_topic = '/sim/odom' # Odometry

        particle_poses_topic = '/pf/particle_poses' # PoseArray
        average_pose_topic = '/pf/avg_pose' # PoseWithCovariance

        rospy.Subscriber(gt_odom_topic, Odometry, self.gt_odom_callback)
        rospy.Subscriber(sim_odom_topic, Odometry, self.sim_odom_callback)
        rospy.Subscriber(particle_poses_topic, PoseArray, self.pose_array_callback)
        rospy.Subscriber(average_pose_topic, PoseWithCovarianceStamped, self.pose_w_cov_callback)

        dir_name = ('results_' + str(time.gmtime().tm_year) + '_' + str(time.gmtime().tm_mon) + '_' + str(time.gmtime().tm_mday) + '___'
                    + str(time.gmtime().tm_hour) + '_' + str(time.gmtime().tm_min) + '_' + str(time.gmtime().tm_sec) + '/')
        if root_folder[-1] != '/':
            dir_name = '/' + dir_name

        storage_path = root_folder + dir_name
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)

        self.odom_file  = open(storage_path + 'true_pose.csv',      "w")
        self.pose_file  = open(storage_path + 'measured_pose.csv',  "w")
        self.array_file = open(storage_path + 'particle_poses.csv', "w")

        self.odom_file.write ('time, pos.x, pos.y, pos.z, quat.x, quat.y, quat.z, quat.w \n')
        self.pose_file.write ('time, pos.x, pos.y, pos.z, quat.x, quat.y, quat.z, quat.w \n')
        self.array_file.write('time, pos.x, pos.y, pos.z, quat.x, quat.y, quat.z, quat.w, idx \n')


    def gt_odom_callback(self, msg):
        time_ = (str(msg.header.stamp.secs) + '.' + str(msg.header.stamp.nsecs))
        pose_ = (str(msg.pose.pose.position.x) + ', ' + str(msg.pose.pose.position.y)
                                                     + ', ' + str(msg.pose.pose.position.z))
        quat_ = (str(msg.pose.pose.orientation.x) + ', ' + str(msg.pose.pose.orientation.y)
                     + ', ' + str(msg.pose.pose.orientation.z) + ', ' + str(msg.pose.pose.orientation.w))

        self.odom_file.write(time_  + ', ' + pose_ + ', ' + quat_ + '\n')



    def sim_odom_callback(self, msg):
        time_ = (str(msg.header.stamp.secs) + '.' + str(msg.header.stamp.nsecs))
        pose_ = (str(msg.pose.pose.position.x) + ', ' + str(msg.pose.pose.position.y)
                                                     + ', ' + str(msg.pose.pose.position.z))
        quat_ = (str(msg.pose.pose.orientation.x) + ', ' + str(msg.pose.pose.orientation.y)
                     + ', ' + str(msg.pose.pose.orientation.z) + ', ' + str(msg.pose.pose.orientation.w))

        self.odom_file.write(time_  + ', ' + pose_ + ', ' + quat_ + '\n')



    def pose_w_cov_callback(self, msg):
        time_ = (str(msg.header.stamp.secs) + '.' + str(msg.header.stamp.nsecs))
        pose_ = (str(msg.pose.pose.position.x) + ', ' + str(msg.pose.pose.position.y)
                                                     + ', ' + str(msg.pose.pose.position.z))
        quat_ = (str(msg.pose.pose.orientation.x) + ', ' + str(msg.pose.pose.orientation.y)
                     + ', ' + str(msg.pose.pose.orientation.z) + ', ' + str(msg.pose.pose.orientation.w))

        self.pose_file.write(time_  + ', ' + pose_ + ', ' + quat_ + '\n')



    def pose_array_callback(self, msg):
        # seq_  = ('seq: '  + str(msg.header.seq) + '\n')
        time_ = (str(msg.header.stamp.secs) + '.' + str(msg.header.stamp.nsecs))
        all_info = ''
        for idx, pose in enumerate(msg.poses):
            pose_ = (str(pose.position.x) + ', ' + str(pose.position.y)
                                        + ', ' + str(pose.position.z))
            quat_ = (str(pose.orientation.x) + ', ' + str(pose.orientation.y)
            + ', ' + str(pose.orientation.z) + ', ' + str(pose.orientation.w))
            all_info += (time_  + ', ' + pose_ + ', ' + quat_ + ', ' + str(idx) + '\n')

        self.array_file.write(all_info)


if __name__ == '__main__':

    rospy.init_node('auv_localization_recorder')
    try:
        AUV_Recorder()
        print("PF results recorder running")
    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch recorder")

    rospy.spin()
