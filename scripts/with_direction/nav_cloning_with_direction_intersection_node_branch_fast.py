#!/usr/bin/env python3
from __future__ import print_function

from numpy import dtype
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# from nav_cloning_with_direction_net_branch import *
from nav_cloning_with_direction_net_branch_fast import *
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
        self.mode = rospy.get_param("/nav_cloning_node/mode", "selected_training")
        self.action_num = 1
        self.dl = deep_learning(n_action = self.action_num)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera_center/image_raw", Image, self.callback)
        self.image_left_sub = rospy.Subscriber("/camera_left/image_raw", Image, self.callback_left_camera)
        self.image_right_sub = rospy.Subscriber("/camera_right/image_raw", Image, self.callback_right_camera)
        self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)
        self.action_pub = rospy.Publisher("action", Int8, queue_size=1)
        self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.srv = rospy.Service('/training', SetBool, self.callback_dl_training)
        self.loop_count_srv = rospy.Service('loop_count',SetBool,self.loop_count_callback)
        self.mode_save_srv = rospy.Service('/model_save', Trigger, self.callback_model_save)
        self.pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.callback_pose)
        self.path_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.callback_path) 
        # self.path_sub = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.callback_path)
 #       self.cmd_dir_sub = rospy.Subscriber("/cmd_dir_intersection", Int8MultiArray, self.callback_cmd,queue_size=1)
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
        self.place = 'cit3f'
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/result_with_dir_'+str(self.mode)+'/'
        self.save_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/model_with_dir_'+str(self.mode)+'/cit3f/branch/'
        self.save_image_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/dataset_with_dir_' + str(self.mode) + '/' + str(self.place) + '/' + str(self.start_time) + '/image/'
        self.save_dir_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/dataset_with_dir_' + str(self.mode) + '/' + str(self.place) + '/' + str(self.start_time) + '/dir/'
        self.save_vel_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/dataset_with_dir_' + str(self.mode) + '/' + str(self.place) + '/' + str(self.start_time) + '/vel/'
        self.load_path =roslib.packages.get_pkg_dir('nav_cloning') + '/data/model_with_dir_'+str(self.mode)+'/cit3f/branch/off_mask/1/model.pt'
        self.load_image_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/dataset_with_dir_' + str(self.mode) + '/' + str(self.place) + '/' + 'old10000' + '/image' + '/image.pt'
        self.load_dir_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/dataset_with_dir_' + str(self.mode) + '/' + str(self.place) + '/' + 'old10000' + '/dir' + '/dir.pt'
        self.load_vel_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/dataset_with_dir_' + str(self.mode) + '/' + str(self.place) + '/' + 'old10000' + '/vel' + '/vel.pt'
        # self.load_path= '/home/rdclab/catkin_ws/src/nav_cloning/data/model_with_dir_selected_training/pytorch/v2_test120000/model_gpu.pt'
        #self.load_path= '/home/rdclab/catkin_ws/src/nav_cloning/data/model_with_dir_selected_training/pytorch/off_new/model_gpu.pt'
        # self.load_path= '/home/rdclab/catkin_ws/src/nav_cloning/data/model_with_dir_selected_training/pytorch/off_branch/model_gpu.pt'
        
        self.previous_reset_time = 0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_the = 0.0
        self.is_started = False
        self.cmd_dir_data = [0, 0, 0]
        self.episode_num = 10000
        self.train_flag = False
        print(self.episode_num)
        self.start_time_s = rospy.get_time()
        os.makedirs(self.path + self.start_time)

        with open(self.path + self.start_time + '/' +  'training.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['step', 'mode', 'loss', 'angle_error(rad)', 'distance(m)','x(m)','y(m)', 'the(rad)', 'direction'])
        self.tracker_sub = rospy.Subscriber("/tracker", Odometry, self.callback_tracker)

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_left_camera(self, data):
        try:
            self.cv_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_right_camera(self, data):
        try:
            self.cv_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_tracker(self, data):
        self.pos_x = data.pose.pose.position.x
        self.pos_y = data.pose.pose.position.y
        rot = data.pose.pose.orientation
        angle = tf.transformations.euler_from_quaternion((rot.x, rot.y, rot.z, rot.w))
        self.pos_the = angle[2]

    def callback_path(self, data):
        self.path_pose = data

    def callback_pose(self, data):
        distance_list = []
        pos = data.pose.pose.position
        for pose in self.path_pose.poses:
            path = pose.pose.position
            distance = np.sqrt(abs((pos.x - path.x)**2 + (pos.y - path.y)**2))
            distance_list.append(distance)

        if distance_list:
            self.min_distance = min(distance_list)

    def callback_cmd(self, data):
        self.cmd_dir_data = data.cmd_dir
        # self.cmd_dir_data = (1, 0, 0)
        # self.cmd_dir_data = (0, 1, 0)
        # self.cmd_dir_data = (0, 0, 1)

    def callback_vel(self, data):
        self.vel = data
        self.action = self.vel.angular.z

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
        if self.cv_image.size != 640 * 480 * 3:
            return
        if self.cv_left_image.size != 640 * 480 * 3:
            return
        if self.cv_right_image.size != 640 * 480 * 3:
            return
        if self.vel.linear.x != 0:
            self.is_started = True
        if self.is_started == False:
            return
        img = resize(self.cv_image, (48, 64), mode='constant')
        
        # r, g, b = cv2.split(img)
        # img = np.asanyarray([r,g,b])

        img_left = resize(self.cv_left_image, (48, 64), mode='constant')
        #r, g, b = cv2.split(img_left)
        #img_left = np.asanyarray([r,g,b])

        img_right = resize(self.cv_right_image, (48, 64), mode='constant')
        #r, g, b = cv2.split(img_right)
        #img_right = np.asanyarray([r,g,b])
        # cmd_dir = np.asanyarray(self.cmd_dir_data)
        ros_time = str(rospy.Time.now())

        if self.episode == 0:
            self.learning = False
            dataset = self.dl.load_dataset(self.load_image_path, self.load_dir_path, self.load_vel_path)
            self.dl.load(self.load_path)            
            print("load model",self.load_path)
        
        if self.episode == self.episode_num:
        #     self.learning = False
        #     self.dl.save(self.save_path)
        #     x_cat, c_cat, t_cat = self.dl.call_dataset()
        #     self.dl.save_tensor(x_cat, self.save_image_path, '/image.pt')
        #     self.dl.save_tensor(c_cat, self.save_dir_path, '/dir.pt')
        #     self.dl.save_tensor(t_cat, self.save_vel_path, '/vel.pt')
        # willow
        # if self.episode == self.episode_num + 1800:
        # cross
        # if self.episode == self.episode_num + 400:
            os.system('killall roslaunch')
            sys.exit()
        if self.episode == self.episode_num + 10000:
            os.system('killall roslaunch')
            sys.exit()

        if self.learning:
            target_action = self.action
            distance = self.min_distance
            
            if self.mode == "selected_training":
                action = self.dl.act(img, self.cmd_dir_data)
                angle_error = abs(action - target_action)
                loss = 0

                if angle_error > 0.05:
                    dataset , dataset_num, train_dataset = self.dl.make_dataset(img,self.cmd_dir_data,target_action)
                    action, loss = self.dl.act_and_trains(img, self.cmd_dir_data, train_dataset)
                    action = action * 1.5
                    action = max(min(action, 0.4), -0.4)

                    if abs(target_action) < 0.1: #0.1
                        dataset , dataset_num, train_dataset = self.dl.make_dataset(img_left,self.cmd_dir_data,target_action-0.2)
                        action_left,  loss_left  = self.dl.act_and_trains(img_left, self.cmd_dir_data, train_dataset)
                        dataset , dataset_num, train_dataset = self.dl.make_dataset(img_right,self.cmd_dir_data,target_action+0.2)
                        action_right, loss_right = self.dl.act_and_trains(img_right, self.cmd_dir_data, train_dataset)
                                
                else:
                    loss = self.dl.trains(2)
                    print("Online Training")

                if self.loop_count_flag:
                    print("loop count")
                    self.vel.linear.x = 0.0
                    self.vel.angular.z = 0.0
                    self.nav_pub.publish(self.vel)
                    self.learning = False
                    self.dl.save(self.save_path)
                    x_cat, c_cat, t_cat = self.dl.call_dataset()
                    self.dl.save_tensor(x_cat, self.save_image_path, '/image.pt')
                    self.dl.save_tensor(c_cat, self.save_dir_path, '/dir.pt')
                    self.dl.save_tensor(t_cat, self.save_vel_path, '/vel.pt')                  
                else:
                    pass

                if self.cmd_dir_data == (0, 1, 0) or self.cmd_dir_data == (0, 0, 1):
                    self.select_dl = False
                else:
                    pass
                        
                if distance >= 0.145 or angle_error > 0.4:
                    self.select_dl = False
                elif distance <= 0.1:
                    self.select_dl = True
                if self.select_dl and self.episode >= 0:
                    target_action = action

            # end mode
            print(str(self.episode) + ", training, loss: " + str(loss) + ", angle_error: " + str(angle_error) + ", distance: " + str(distance) + ", self.cmd_dir_data: " + str(self.cmd_dir_data))
            self.episode += 1
            line = [str(self.episode), "training", str(loss), str(angle_error), str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the), str(self.cmd_dir_data)]
            with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(line)
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
            self.nav_pub.publish(self.vel)

        else:
            #print('\033[32m'+'test_mode'+'\033[0m')
            target_action = self.dl.act(img, self.cmd_dir_data)
            if abs(target_action) >1.82:
                target_action=1.82
            else:
                pass
            distance = self.min_distance
            print(str(self.episode) + ", test, angular:" + str(target_action) + ", distance: " + str(distance) + ", self.cmd_dir_data: " + str(self.cmd_dir_data))

            self.episode += 1
            angle_error = abs(self.action - target_action)
            line = [str(self.episode), "test", "0", str(angle_error), str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the), str(self.cmd_dir_data)]
            with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(line)
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
            self.nav_pub.publish(self.vel)

        # temp = copy.deepcopy(img)
        # cv2.imshow("Resized Image", temp)
        # temp = copy.deepcopy(img_left)
        # cv2.imshow("Resized Left Image", temp)
        # temp = copy.deepcopy(img_right)
        # cv2.imshow("Resized Right Image", temp)
        # cv2.waitKey(1)

if __name__ == '__main__':
    rg = nav_cloning_node()
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()