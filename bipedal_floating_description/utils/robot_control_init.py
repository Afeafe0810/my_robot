import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray 

from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ContactsState

from nav_msgs.msg import Odometry

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

import qpsolvers

import pink
from pink import solve_ik
from pink.tasks import FrameTask, JointCouplingTask, PostureTask
import meshcat_shapes
import qpsolvers

import numpy as np
np.set_printoptions(precision=2)

from sys import argv
from os.path import dirname, join, abspath
import os
from copy import deepcopy
from math import cos, sin, cosh, sinh, pi, sqrt
from scipy.spatial.transform import Rotation

import pandas as pd
import csv


class ULC_init:
    def __init__(self):
        
        self.sub = {
            'p_base_in_wf': None,
            'r_base2wf': None,
            'state': None,
            'jp': None,
        }
               
    @staticmethod
    def create_publishers(node: Node):
        '''Effort publisher 是 ROS2-control 的力矩，負責控制各個關節的力矩'''
        
        return {
            'effort': node.create_publisher(Float64MultiArray, '/effort_controllers/commands', 10),  # 只有這個負責控制，其他只是用來可視化
            
            'position': node.create_publisher(Float64MultiArray, '/position_controller/commands', 10),
            'velocity': node.create_publisher(Float64MultiArray, '/velocity_controller/commands', 10),
            'velocity_cmd': node.create_publisher(Float64MultiArray, '/velocity_command/commands', 10),
            'l_gravity': node.create_publisher(Float64MultiArray, '/l_gravity', 10),
            'r_gravity': node.create_publisher(Float64MultiArray, '/r_gravity', 10),
            'alip_x': node.create_publisher(Float64MultiArray, '/alip_x_data', 10),
            'alip_y': node.create_publisher(Float64MultiArray, '/alip_y_data', 10),
            'torque_l': node.create_publisher(Float64MultiArray, '/torqueL_data', 10),
            'torque_r': node.create_publisher(Float64MultiArray, '/torqueR_data', 10),
            'ref': node.create_publisher(Float64MultiArray, '/ref_data', 10),
            'joint_trajectory_controller': node.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10),
            'pel': node.create_publisher(Float64MultiArray, '/px_data', 10),
            'com': node.create_publisher(Float64MultiArray, '/com_data', 10),
            'lf': node.create_publisher(Float64MultiArray, '/lx_data', 10),
            'rf': node.create_publisher(Float64MultiArray, '/rx_data', 10),
        }
    
    def create_subscribers(self, node: Node):
        return {
            'base' : node.create_subscription(Odometry, '/odom', self.base_in_wf_callback, 10), #base_state_subscribe
            'state' : node.create_subscription(Float64MultiArray, 'state_topic', self.state_callback, 10),
            'joint_states' : node.create_subscription(JointState, '/joint_states', self.joint_states_callback, 10), #joint_state_subscribe
        }
        
    @staticmethod
    def base_in_wf_callback(self, msg):
        ''' 訂閱base_in_wf的位置和旋轉矩陣'''
        base = msg.pose.pose.position
        quaters_base = msg.pose.pose.orientation ##四元數法
        quaters_base = Rotation.from_quat([quaters_base.x, quaters_base.y, quaters_base.z, quaters_base.w])
        self.sub['p_base_in_wf'] = np.vstack(( base.x, base.y, base.z ))
        self.sub['r_base2wf'] = quaters_base.as_matrix()
        # print('base\n',sub['p_base_in_wf'].flatten())
    
    def state_callback(self, msg):
        """ 接收我們手動pub出的state """
        self.sub['state'] = msg.data[0]
    
    @staticmethod    
    def joint_states_callback(msg):
        '''把訂閱到的關節位置、差分與飽和限制算出速度,並轉成我們想要的順序'''
        nonlocal callcount
        callcount += 1
                                                            
        if len(msg.position) == 12: # 將關節順序轉成我們想要的
            jp_dict = {joint:value for joint,value in zip(original_joint_order, msg.position)}
            sub['jp'] =  np.vstack([ jp_dict[joint] for joint in desired_joint_order ])
            
        if callcount == 5:
            #========================把sub的資料全部深複製成self的property,使得跑main_callback的時候不會中途被改變==================================#
            self.pt.p_base_in_wf = deepcopy(sub['p_base_in_wf'])
            self.pt.r_base2wf = deepcopy(sub['r_base2wf'])
            self.state = deepcopy(sub['state'])
            self.jp = jpfilter.send(deepcopy(sub['jp']))
            #========================把5次的點差分出速度,加上飽和條件與濾波==================================#
            self.jv = jvfilter.send( jpdiff.send(self.jp) )
            self.jv = np.maximum( self.jv, -0.75 )
            self.jv = np.minimum( self.jv,  0.75 )
            
            print('jp:',self.jp.flatten()*180/pi)
            print('jv:',self.jv.flatten()*180/pi)
            
            #==========================================================#
            self.main_callback()
            callcount = 0 
                
def subscriber_create():
            callcount=0 
            #========================存放訂閱的資料,每5次輸出給self==================================#
            sub = {
                
            
            
            
                    
            
                       
            