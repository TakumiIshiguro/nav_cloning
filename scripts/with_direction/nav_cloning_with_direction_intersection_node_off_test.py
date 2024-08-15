#!/usr/bin/env python3
from __future__ import print_function

from numpy import dtype
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# from nav_cloning_with_direction_net import *
from nav_cloning_with_direction_net_off import *
from skimage.transform import resize
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int8
from std_srvs.srv import Trigger
from nav_msgs.msg import Path
from std_msgs.msg import Int8MultiArray
#from waypoint_nav.msg import cmd_dir_intersection
from scenario_navigation_msgs.msg import cmd_dir_intersection
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_srvs.srv import Empty
from std_srvs.srv import SetBool, SetBoolResponse
import csv
import os
import time
import copy
import sys
import tf
from nav_msgs.msg import Odometry

class nav_cloning_node:
    def __init__(self):
        rospy.init_node('nav_cloning_node', anonymous=True)
        self.mode = rospy.get_param("/nav_cloning_node/mode", "use_dl_output")
        self.action_num = 1
        self.dl = deep_learning(n_action = self.action_num)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.action_pub = rospy.Publisher("action", Int8, queue_size=1)
        self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.srv = rospy.Service('/training', SetBool, self.callback_dl_training)
        self.loop_count_srv = rospy.Service('loop_count',SetBool,self.loop_count_callback)
        self.mode_save_srv = rospy.Service('/model_save', Trigger, self.callback_model_save) 
        self.cmd_dir_sub = rospy.Subscriber("/cmd_dir_intersection", cmd_dir_intersection, self.callback_cmd,queue_size=1)
        self.min_distance = 0.0
        self.action = 0.0
        self.episode = 0
        self.vel = Twist()
        self.path_pose = PoseArray()
        self.cv_image = np.zeros((480,640,3), np.uint8)
        self.cv_left_image = np.zeros((480,640,3), np.uint8)
        self.cv_right_image = np.zeros((480,640,3), np.uint8)
        self.learning = True
        self.select_dl = False
        self.loop_count_flag = False
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/result_with_dir_'+str(self.mode)+'/'
        self.save_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/model_with_dir_'+str(self.mode)+'/pytorch/'
        self.load_path='/home/takumi/catkin_ws/src/nav_cloning/data/model_with_dir_selected_training/pytorch/off_pad_3_loop_1_30_ep_add/model_gpu.pt'
        # self.load_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/model_with_dir_'+str(self.mode)+'/pytorch/off_pad_3_loop_1_30_ep_add/model_gpu.pt'

        self.previous_reset_time = 0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_the = 0.0
        self.is_started = False
        # self.cmd_dir_data = (0, 0, 0)
        self.episode_num =12000
        self.target_dataset = 8500
        self.train_flag = False
        self.padding_data = 3
        print(self.episode_num)
        #self.cmd_dir_data = [0, 0, 0]
        self.start_time_s = rospy.get_time()
        os.makedirs(self.path + self.start_time)

        with open(self.path + self.start_time + '/' +  'training.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['step', 'mode', 'loss', 'angle_error(rad)', 'distance(m)','x(m)','y(m)', 'the(rad)', 'direction'])

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_cmd(self, data):
        self.cmd_dir_data = data.cmd_dir

    def callback_dl_training(self, data):
        resp = SetBoolResponse()
        self.learning = data.data
        resp.message = "Training: " + str(self.learning)
        resp.success = True
        return resp

    def loop_count_callback(self,data):
        resp = SetBoolResponse()
        self.loop_count_flag = data.data
        resp.message = "count flag"
        resp.success= True
        return resp

    def callback_model_save(self, data):
        model_res = SetBoolResponse()
        self.dl.save(self.save_path)
        model_res.message ="model_save"
        model_res.success = True
        return model_res
    
    def loop(self):
        # print('\033[32m'+'test_mode'+'\033[0m')
        if self.cv_image.size != 640 * 480 * 3:
            return
        if self.cv_left_image.size != 640 * 480 * 3:
            return
        if self.cv_right_image.size != 640 * 480 * 3:
            return
        img = resize(self.cv_image, (48, 64), mode='constant')
        ros_time = str(rospy.Time.now())

        if self.episode == 0:
            self.learning = False
            #self.dl.save(self.save_path)
            self.dl.load(self.load_path)
            print("load model",self.load_path)
        
        print('\033[32m'+'test_mode'+'\033[0m')
        # stop
        if self.cmd_dir_data == (0,0,0):
            print('\033[32m'+'stop'+'\033[0m')
            self.vel.linear.x = 0.0
            self.vel.angular.z = 0.0
            target_action = 0.0
        else:
            print('\033[32m'+'move'+'\033[0m')
            target_action = self.dl.act(img, self.cmd_dir_data)
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
            if abs(target_action) >1.82:
                target_action=1.82
            else:
                pass

        print(str(self.episode) + ", test, angular:" + str(target_action) + ", self.cmd_dir_data: " + str(self.cmd_dir_data))

        self.episode += 1
        self.nav_pub.publish(self.vel)

if __name__ == '__main__':
    rg = nav_cloning_node()
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()
