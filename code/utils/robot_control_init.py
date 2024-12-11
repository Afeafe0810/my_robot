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
import copy
import math
from scipy.spatial.transform import Rotation as R


class ULC_init:
    
    @staticmethod  
    def create_publishers(node: Node):
        '''effort publisher是ROS2-control的力矩, 負責控制各個關節的力矩->我們程式的目的就是為了pub他'''
        return {
            #只有effort才是真正的控制命令，其他只是用來追蹤數據
            "effort" : node.create_publisher(Float64MultiArray , '/effort_controllers/commands', 10),
            
            "position" : node.create_publisher(Float64MultiArray , '/position_controller/commands', 10),
            "velocity" : node.create_publisher(Float64MultiArray , '/velocity_controller/commands', 10),
            "vcmd" :  node.create_publisher(Float64MultiArray , '/velocity_command/commands', 10),
            "gravity_l" :  node.create_publisher(Float64MultiArray , '/l_gravity', 10),
            "gravity_r" : node.create_publisher(Float64MultiArray , '/r_gravity', 10),
            "alip_x" : node.create_publisher(Float64MultiArray , '/alip_x_data', 10),
            "alip_y" : node.create_publisher(Float64MultiArray , '/alip_y_data', 10),
            "torque_l" : node.create_publisher(Float64MultiArray , '/torqueL_data', 10),
            "torque_r" : node.create_publisher(Float64MultiArray , '/torqueR_data', 10),
            "ref" : node.create_publisher(Float64MultiArray , '/ref_data', 10),
            "joint_trajectory_controller" : node.create_publisher(
                JointTrajectory , '/joint_trajectory_controller/joint_trajectory', 10
            ),
            "pel" : node.create_publisher(Float64MultiArray , '/px_data', 10),
            "com" : node.create_publisher(Float64MultiArray , '/com_data', 10),
            "lf" : node.create_publisher(Float64MultiArray , '/lx_data', 10),
            "rf" : node.create_publisher(Float64MultiArray , '/rx_data', 10),
        }
        
    @classmethod
    def create_subscribers(cls, node: Node):
        '''主要是為了訂閱base, joint_states, state'''
        return{
            "base": node.create_subscription(
                Odometry, '/odom', 
                lambda msg: cls.__base_in_wf(node, msg), 
                10
            ),
            
            "joint_states": node.create_subscription(
                JointState, '/joint_states',
                lambda msg : cls.__joint_states_callback(node, msg),
                10
            ),
            
            "state": node.create_subscription(
                Float64MultiArray, 'state_topic',
                lambda msg : cls.__state_callback(node, msg),
                10
            ),
        
            "lf_contact": node.create_subscription(
                ContactsState, '/l_foot/bumper_demo',
                lambda msg: cls.__contact_callback(node, msg),
                10
            ),
        
            "rf_contact": node.create_subscription(
                ContactsState, '/r_foot/bumper_demo',
                lambda msg: cls.__contact_callback(node, msg),
                10
            ),
        }

    @staticmethod
    def loadMeshcatModel(urdf_path):
        robot = pin.RobotWrapper.BuildFromURDF(
            filename = urdf_path,
            package_dirs = ["."],
            root_joint=None,
        )
        print(f"URDF description successfully loaded in {robot}")
        return robot

    @staticmethod
    def load_URDF(node, urdf_path):
        robot = pin.RobotWrapper.BuildFromURDF(
            filename=urdf_path,
            package_dirs=["."],
            # root_joint=pin.JointModelFreeFlyer(),
            root_joint=None,
        )
        
        print(f"URDF description successfully loaded in {robot}")

        #從骨盆建下來的模擬模型
        pinocchio_model_dir = "/home/ldsc/ros2_ws/src"
        urdf_filename = pinocchio_model_dir + '/bipedal_floating_description/urdf/bipedal_floating.xacro' if len(argv)<2 else argv[1]
        # Load the urdf model
        node.bipedal_floating_model  = pin.buildModelFromUrdf(urdf_filename)
        print('model name: ' + node.bipedal_floating_model.name)
        # Create data required by the algorithms
        node.bipedal_floating_data = node.bipedal_floating_model.createData()

        #左單支撐腳
        pinocchio_model_dir = "/home/ldsc/ros2_ws/src"
        urdf_filename = pinocchio_model_dir + '/bipedal_floating_description/urdf/stance_l.xacro' if len(argv)<2 else argv[1]
        # Load the urdf model
        node.stance_l_model  = pin.buildModelFromUrdf(urdf_filename)
        print('model name: ' + node.stance_l_model.name)
        # Create data required by the algorithms
        node.stance_l_data = node.stance_l_model.createData()

        #右單支撐腳
        pinocchio_model_dir = "/home/ldsc/ros2_ws/src"
        urdf_filename = pinocchio_model_dir + '/bipedal_floating_description/urdf/stance_r_gravity.xacro' if len(argv)<2 else argv[1]
        # Load the urdf model
        node.stance_r_model  = pin.buildModelFromUrdf(urdf_filename)
        print('model name: ' + node.stance_r_model.name)
        # Create data required by the algorithms
        node.stance_r_data = node.stance_r_model.createData()

        #雙足模型_以左腳建起
        pinocchio_model_dir = "/home/ldsc/ros2_ws/src"
        urdf_filename = pinocchio_model_dir + '/bipedal_floating_description/urdf/bipedal_l_gravity.xacro' if len(argv)<2 else argv[1]
        # Load the urdf model
        node.bipedal_l_model  = pin.buildModelFromUrdf(urdf_filename)
        print('model name: ' + node.bipedal_l_model.name)
        # Create data required by the algorithms
        node.bipedal_l_data = node.bipedal_l_model.createData()

        #雙足模型_以右腳建起
        pinocchio_model_dir = "/home/ldsc/ros2_ws/src"
        urdf_filename = pinocchio_model_dir + '/bipedal_floating_description/urdf/bipedal_r_gravity.xacro' if len(argv)<2 else argv[1]
        # Load the urdf model
        node.bipedal_r_model  = pin.buildModelFromUrdf(urdf_filename)
        print('model name: ' + node.bipedal_r_model.name)
        # Create data required by the algorithms
        node.bipedal_r_data = node.bipedal_r_model.createData()

        return robot
      
    @staticmethod
    def __base_in_wf(node, msg:Odometry):
        P_base_x = msg.pose.pose.position.x
        P_base_y = msg.pose.pose.position.y
        P_base_z = msg.pose.pose.position.z
        node.P_B_wf = np.array([[P_base_x],[P_base_y],[P_base_z]])

        O_base_x = msg.pose.pose.orientation.x
        O_base_y = msg.pose.pose.orientation.y
        O_base_z = msg.pose.pose.orientation.z
        O_base_w = msg.pose.pose.orientation.w
        base_quaternions = R.from_quat([O_base_x, O_base_y, O_base_z, O_base_w])
        node.O_wfB = base_quaternions.as_matrix()  #注意

    @staticmethod
    def __joint_states_callback(node, msg:JointState ):
        
        # Original ndarray order
        original_order = [
            'L_Hip_Yaw', 'L_Hip_Pitch', 'L_Knee_Pitch', 'L_Ankle_Pitch', 
            'L_Ankle_Roll', 'R_Hip_Roll', 'R_Hip_Yaw', 'R_Knee_Pitch', 
            'R_Hip_Pitch', 'R_Ankle_Pitch', 'L_Hip_Roll', 'R_Ankle_Roll'
        ]

        # Desired order
        desired_order = [
            'L_Hip_Roll', 'L_Hip_Yaw', 'L_Hip_Pitch', 'L_Knee_Pitch', 
            'L_Ankle_Pitch', 'L_Ankle_Roll', 'R_Hip_Roll', 'R_Hip_Yaw', 
            'R_Hip_Pitch', 'R_Knee_Pitch', 'R_Ankle_Pitch', 'R_Ankle_Roll'
        ]

        if len(msg.velocity) == 12:
            velocity_order_dict = {joint: value for joint, value in zip(original_order, np.array(msg.velocity))}

            node.jv_sub = np.array([velocity_order_dict[joint] for joint in desired_order])

        if len(msg.position) == 12:
            position_order_dict = {joint: value for joint, value in zip(original_order, np.array(msg.position))}
            node.jp_sub = np.array([position_order_dict[joint] for joint in desired_order])

        node.call += 1
        if node.call == 5:
            node.main_controller_callback()
            node.call = 0
    
    @staticmethod
    def __state_callback(node,msg:Float64MultiArray):
        
        node.pub_state = msg.data[0]
        node.state_collect()
    
    @staticmethod
    def __contact_callback(node:Node, msg:ContactsState ):
        if msg.header.frame_id == 'l_foot_1':
            if len(msg.states)>=1:
                node.l_contact = 1
            else:
                node.l_contact = 0
        elif msg.header.frame_id == 'r_foot_1':
            if len(msg.states)>=1:
                node.r_contact = 1
            else:
                node.r_contact = 0

