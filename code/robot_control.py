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

import numpy as np; np.set_printoptions(precision=2)

from sys import argv
from os.path import dirname, join, abspath
import os
import copy
import math
from scipy.spatial.transform import Rotation as R

import pandas as pd
import csv

from linkattacher_msgs.srv import AttachLink
from linkattacher_msgs.srv import DetachLink

#================ import other code =====================#
from utils.ros_interfaces import ROSInterfaces
from utils.rc_frame_kinermatic import RobotFrame
from utils.robot_control_traj import *
from utils.robot_control_knee_control import *
#========================================================#

class UpperLevelController(Node):

    def __init__(self):
        #==============================================================node==============================================================#
        super().__init__('upper_level_controllers')
        self.ros = ROSInterfaces(self, self.main_controller_callback)
        #==============================================================robot interface==============================================================#
        
        
        self.robot = self.load_URDF("/home/ldsc/ros2_ws/src/bipedal_floating_description/urdf/bipedal_floating.pin.urdf")
        
        # Initialize meschcat visualizer
        self.viz = pin.visualize.MeshcatVisualizer(
            self.robot.model, self.robot.collision_model, self.robot.visual_model
        )
        self.robot.setVisualizer(self.viz, init=False)
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel()

        # Set initial robot configuration
        print(self.robot.model)
        print(self.robot.q0)
        self.init_configuration = pink.Configuration(self.robot.model, self.robot.data, self.robot.q0)
        self.viz.display(self.init_configuration.q)
        
        # Tasks initialization for IK
        self.tasks = self.tasks_init()
        
        #==============================================================robot constant==============================================================#     
        self.call = 0
        self.timer_period = 0.01 # seconds 跟joint state update rate&取樣次數有關
        # self.timer = self.create_timer(self.timer_period, self.main_controller_callback)

        self.stance = 2
        self.stance_past = 2
        self.DS_time = 0.0
        self.RSS_time = 0.0
        self.LSS_time = 0.0
        self.RSS_count = 0
        self.DDT = 2
        self.RDT = 1
        self.LDT = 1

        #==============================================================robot frame==============================================================#     
        #data in wf_initial_data
        self.P_B_wf = np.zeros((3,1))
        self.P_PV_wf = np.array([[0.0],[0.0],[0.6]])
        self.P_COM_wf = np.array([[0.0],[0.0],[0.6]])
        self.P_L_wf= np.array([[0.0],[0.1],[0.0]])
        self.P_R_wf = np.array([[0.0],[-0.1],[0.0]])
        
        self.O_wfB = np.zeros((3,3))
        self.O_wfPV = np.zeros((3,3))
        self.O_wfL = np.zeros((3,3))
        self.O_wfR = np.zeros((3,3))

        self.state_past = 0
        #data_in_pf 

        #ALIP
        #time
        self.contact_t = 0.0
        self.alip_t = 0.0
        #online_planning
        self.P_cf_wf = np.zeros((3,1))
        self.X0 = np.zeros((2,1))
        self.Y0 = np.zeros((2,1))
        self.Psw2com_0 = np.zeros((2,1))
        
        #--torque
        self.ap_L = 0.0
        self.ap_past_L = 0.0
        self.ar_L = 0.0
        self.ar_past_L = 0.0
        self.ap_R = 0.0
        self.ap_past_R = 0.0
        self.ar_R = 0.0
        self.ar_past_R = 0.0

        #==============================================================ref==============================================================#     
        self.ref_x_L = np.zeros((2,1))
        self.ref_y_L = np.zeros((2,1))
        self.ref_x_R = np.zeros((2,1))
        self.ref_y_R = np.zeros((2,1))

        #touch for am tracking check
        self.touch = 0
 
        # Initialize the service client
        self.attach_link_client = self.create_client(AttachLink, '/ATTACHLINK')
        self.detach_link_client = self.create_client(DetachLink, '/DETACHLINK')
        
        #==============================================================逐步修改==============================================================#
        self.frame = RobotFrame() # 各部位的位置與姿態
        
    
    def attach_links(self, model1_name, link1_name, model2_name, link2_name):
        req = AttachLink.Request()
        req.model1_name = model1_name
        req.link1_name = link1_name
        req.model2_name = model2_name
        req.link2_name = link2_name

        self.future = self.attach_link_client.call_async(req)

    def detach_links(self, model1_name, link1_name, model2_name, link2_name):
        req = DetachLink.Request()
        req.model1_name = model1_name
        req.link1_name = link1_name
        req.model2_name = model2_name
        req.link2_name = link2_name

        self.future = self.detach_link_client.call_async(req)

    def load_URDF(self, urdf_path):
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
        self.bipedal_floating_model  = pin.buildModelFromUrdf(urdf_filename)
        print('model name: ' + self.bipedal_floating_model.name)
        # Create data required by the algorithms
        self.bipedal_floating_data = self.bipedal_floating_model.createData()

        #左單支撐腳
        pinocchio_model_dir = "/home/ldsc/ros2_ws/src"
        urdf_filename = pinocchio_model_dir + '/bipedal_floating_description/urdf/stance_l.xacro' if len(argv)<2 else argv[1]
        # Load the urdf model
        self.stance_l_model  = pin.buildModelFromUrdf(urdf_filename)
        print('model name: ' + self.stance_l_model.name)
        # Create data required by the algorithms
        self.stance_l_data = self.stance_l_model.createData()

        #右單支撐腳
        pinocchio_model_dir = "/home/ldsc/ros2_ws/src"
        urdf_filename = pinocchio_model_dir + '/bipedal_floating_description/urdf/stance_r_gravity.xacro' if len(argv)<2 else argv[1]
        # Load the urdf model
        self.stance_r_model  = pin.buildModelFromUrdf(urdf_filename)
        print('model name: ' + self.stance_r_model.name)
        # Create data required by the algorithms
        self.stance_r_data = self.stance_r_model.createData()

        #雙足模型_以左腳建起
        pinocchio_model_dir = "/home/ldsc/ros2_ws/src"
        urdf_filename = pinocchio_model_dir + '/bipedal_floating_description/urdf/bipedal_l_gravity.xacro' if len(argv)<2 else argv[1]
        # Load the urdf model
        self.bipedal_l_model  = pin.buildModelFromUrdf(urdf_filename)
        print('model name: ' + self.bipedal_l_model.name)
        # Create data required by the algorithms
        self.bipedal_l_data = self.bipedal_l_model.createData()

        #雙足模型_以右腳建起
        pinocchio_model_dir = "/home/ldsc/ros2_ws/src"
        urdf_filename = pinocchio_model_dir + '/bipedal_floating_description/urdf/bipedal_r_gravity.xacro' if len(argv)<2 else argv[1]
        # Load the urdf model
        self.bipedal_r_model  = pin.buildModelFromUrdf(urdf_filename)
        print('model name: ' + self.bipedal_r_model.name)
        # Create data required by the algorithms
        self.bipedal_r_data = self.bipedal_r_model.createData()

        return robot
        
    def tasks_init(self):
        # Tasks initialization for IK
        left_foot_task = FrameTask(
            "l_foot",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        pelvis_task = FrameTask(
            "base_link",
            position_cost=1.0,
            orientation_cost=0.0,
        )
        right_foot_task = FrameTask(
            "r_foot_1",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        posture_task = PostureTask(
            cost=1e-1,  # [cost] / [rad]
        )
        tasks = {
            # 'left_foot_task': left_foot_task,
            'pelvis_task': pelvis_task,
            # 'right_foot_task': right_foot_task,
            'posture_task': posture_task,
        }
        return tasks

    def contact_collect(self):
        '''
            只複製並回傳(l_contact,r_contact)
        '''
        l_contact = copy.deepcopy(self.l_contact)
        r_contact = copy.deepcopy(self.r_contact)
        # print("L:",l_contact,"R:",r_contact)

        return l_contact,r_contact

    def contact_callback(self,msg):
        if msg.header.frame_id == 'l_foot_1':
            if len(msg.states)>=1:
                self.l_contact = 1
            else:
                self.l_contact = 0
        elif msg.header.frame_id == 'r_foot_1':
            if len(msg.states)>=1:
                self.r_contact = 1
            else:
                self.r_contact = 0
        self.contact_collect()

    def state_collect(self):
        self.state_current = copy.deepcopy(self.pub_state)

        return self.state_current
    
    def state_callback(self,msg):
        
        self.pub_state = msg.data[0]
        self.state_collect()
        
    def collect_joint_data(self):
        '''
        就只是收集而已
        '''
        joint_position = copy.deepcopy(self.jp_sub)
        joint_velocity = copy.deepcopy(self.jv_sub)

        joint_position = np.reshape(joint_position,(12,1))
        joint_velocity = np.reshape(joint_velocity,(12,1))

        return joint_position,joint_velocity

    def joint_position_filter(self,joint_position):
        
        jp_sub = copy.deepcopy(joint_position)

        self.jp = 1.1580*self.jp_p - 0.4112*self.jp_pp + 0.1453*self.jp_sub_p + 0.1078*self.jp_sub_pp #10Hz

        self.jp_pp = copy.deepcopy(self.jp_p)
        self.jp_p = copy.deepcopy(self.jp)
        self.jp_sub_pp = copy.deepcopy(self.jp_sub_p)
        self.jp_sub_p = copy.deepcopy(jp_sub)

        return self.jp

    def joint_velocity_cal(self,joint_position):
        '''
        用差分計算速度，並且加上飽和條件[-0.75, 0.75]、更新joint_position_past(感覺沒意義)
        '''
        joint_position_now = copy.deepcopy(joint_position)
        joint_velocity_cal = (joint_position_now - self.joint_position_past)/self.timer_period
        self.joint_position_past = joint_position_now     
        
        joint_velocity_cal = np.reshape(joint_velocity_cal,(12,1))

        for i in range(len(joint_velocity_cal)):
            if joint_velocity_cal[i,0]>= 0.75:
                joint_velocity_cal[i,0] = 0.75
            elif joint_velocity_cal[i,0]<= -0.75:
                joint_velocity_cal[i,0] = -0.75

        return joint_velocity_cal

    def joint_velocity_filter(self,joint_velocity):
        '''
        把差分得到的速度做5個點差分，但pp是什麼意思？？是怎麼濾的
        '''
        jv_sub = copy.deepcopy(joint_velocity)

        # self.jv = 1.1580*self.jv_p - 0.4112*self.jv_pp + 0.1453*self.jv_sub_p + 0.1078*self.jv_sub_pp #10Hz
        # self.jv = 0.5186*self.jv_p - 0.1691*self.jv_pp + 0.4215*self.jv_sub_p + 0.229*self.jv_sub_pp #20Hz
        self.jv = 0.0063*self.jv_p - 0.0001383*self.jv_pp + 1.014*self.jv_sub_p -0.008067*self.jv_sub_pp #100Hz

        self.jv_pp = copy.deepcopy(self.jv_p)
        self.jv_p = copy.deepcopy(self.jv)
        self.jv_sub_pp = copy.deepcopy(self.jv_sub_p)
        self.jv_sub_p = copy.deepcopy(jv_sub)

        return self.jv

    def joint_states_callback(self, msg):
        
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

            self.jv_sub = np.array([velocity_order_dict[joint] for joint in desired_order])

        if len(msg.position) == 12:
            position_order_dict = {joint: value for joint, value in zip(original_order, np.array(msg.position))}
            self.jp_sub = np.array([position_order_dict[joint] for joint in desired_order])

        self.collect_joint_data()

        self.call += 1
        if self.call == 5:
            self.main_controller_callback()
            self.call = 0

    def xyz_rotation(self,axis,theta):
        cos = math.cos
        sin = math.sin
        R = np.array((3,3))
        if axis == 'x':
            R = np.array([[1,0,0],[0,cos(theta),-sin(theta)],[0,sin(theta),cos(theta)]])
        elif axis == 'y':
            R = np.array([[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]])
        elif axis == 'z':
            R = np.array([[cos(theta),-sin(theta),0],[sin(theta),cos(theta),0],[0,0,1]])
        return R    

    def get_position_pf(self,configuration):
        '''
        把各個座標系對pf的原點、旋轉矩陣賦值到self內
        '''
        #frame data in pf
        PV_pf = configuration.get_transform_frame_to_world("pelvis_link")
        # print("p",pelvis)
        Lhr_pf = configuration.get_transform_frame_to_world("l_hip_yaw_1")
        Lhy_pf = configuration.get_transform_frame_to_world("l_hip_pitch_1")
        Lhp_pf = configuration.get_transform_frame_to_world("l_thigh_1")
        Lkp_pf = configuration.get_transform_frame_to_world("l_shank_1")
        Lap_pf = configuration.get_transform_frame_to_world("l_ankle_1")
        Lar_pf = configuration.get_transform_frame_to_world("l_foot_1")
        L_pf = configuration.get_transform_frame_to_world("l_foot")
        # print("l_foot:",self.l_foot.translation)
        Rhr_pf = configuration.get_transform_frame_to_world("r_hip_yaw_1")
        Rhy_pf = configuration.get_transform_frame_to_world("r_hip_pitch_1")
        Rhp_pf = configuration.get_transform_frame_to_world("r_thigh_1")
        Rkp_pf = configuration.get_transform_frame_to_world("r_shank_1")
        Rap_pf = configuration.get_transform_frame_to_world("r_ankle_1")
        Rar_pf = configuration.get_transform_frame_to_world("r_foot_1")
        R_pf = configuration.get_transform_frame_to_world("r_foot")
        # print("r_foot:",self.r_foot.translation)        

        #frame origin position in pf
        self.P_PV_pf = np.reshape(PV_pf.translation,(3,1))

        self.P_Lhr_pf = np.reshape(Lhr_pf.translation,(3,1))
        self.P_Lhy_pf = np.reshape(Lhy_pf.translation,(3,1))
        self.P_Lhp_pf = np.reshape(Lhp_pf.translation,(3,1))
        self.P_Lkp_pf = np.reshape(Lkp_pf.translation,(3,1))
        self.P_Lap_pf = np.reshape(Lap_pf.translation,(3,1))
        self.P_Lar_pf = np.reshape(Lar_pf.translation,(3,1))
        self.P_L_pf = np.reshape(L_pf.translation,(3,1))

        self.P_Rhr_pf = np.reshape(Rhr_pf.translation,(3,1))
        self.P_Rhy_pf = np.reshape(Rhy_pf.translation,(3,1))
        self.P_Rhp_pf = np.reshape(Rhp_pf.translation,(3,1))
        self.P_Rkp_pf = np.reshape(Rkp_pf.translation,(3,1))
        self.P_Rap_pf = np.reshape(Rap_pf.translation,(3,1))
        self.P_Rar_pf = np.reshape(Rar_pf.translation,(3,1))
        self.P_R_pf = np.reshape(R_pf.translation,(3,1))

        #frame orientation in pf
        self.O_pfPV = np.reshape(PV_pf.rotation,(3,3))
        self.O_pfLhr = np.reshape(Lhr_pf.rotation,(3,3))
        self.O_pfLhy = np.reshape(Lhy_pf.rotation,(3,3))
        self.O_pfLhp = np.reshape(Lhp_pf.rotation,(3,3))
        self.O_pfLkp = np.reshape(Lkp_pf.rotation,(3,3))
        self.O_pfLap = np.reshape(Lap_pf.rotation,(3,3))
        self.O_pfLar = np.reshape(Lar_pf.rotation,(3,3))
        self.O_pfL = np.reshape(L_pf.rotation,(3,3))

        self.O_pfRhr = np.reshape(Rhr_pf.rotation,(3,3))
        self.O_pfRhy = np.reshape(Rhy_pf.rotation,(3,3))
        self.O_pfRhp = np.reshape(Rhp_pf.rotation,(3,3))
        self.O_pfRkp = np.reshape(Rkp_pf.rotation,(3,3))
        self.O_pfRap = np.reshape(Rap_pf.rotation,(3,3))
        self.O_pfRar = np.reshape(Rar_pf.rotation,(3,3))
        self.O_pfR = np.reshape(R_pf.rotation,(3,3))
          
    def get_posture(self):
        '''
        回傳(骨盆相對於左腳，骨盆相對於右腳)，但body transfer不知道是什麼
        '''
        cos = math.cos
        sin = math.sin

        pelvis_p = copy.deepcopy(self.P_PV_pf)
        l_foot_p = copy.deepcopy(self.P_L_pf)
        r_foot_p = copy.deepcopy(self.P_R_pf)

        # pelvis_p = copy.deepcopy(self.P_PV_wf)
        # l_foot_p = copy.deepcopy(self.P_L_wf)
        # r_foot_p = copy.deepcopy(self.P_R_wf)

        pelvis_o = copy.deepcopy(self.O_pfPV)
        l_foot_o = copy.deepcopy(self.O_pfL)
        r_foot_o = copy.deepcopy(self.O_pfR)

        # pelvis_o = copy.deepcopy(self.O_wfPV)
        # l_foot_o = copy.deepcopy(self.O_wfL)
        # r_foot_o = copy.deepcopy(self.O_wfR)

        ##////把旋轉矩陣換成歐拉角zyx
        pR = R.from_matrix(pelvis_o).as_euler('zyx', degrees=False)   
        P_Yaw = pR[0]
        P_Pitch = pR[1]
        P_Roll = pR[2]

        lR = R.from_matrix(l_foot_o).as_euler('zyx', degrees=False) 
        L_Yaw = lR[0]
        L_Pitch = lR[1]
        L_Roll = lR[2]
        
        rR = R.from_matrix(r_foot_o).as_euler('zyx', degrees=False) 
        R_Yaw = rR[0]
        R_Pitch = rR[1]
        R_Roll = rR[2]

        self.PX = np.array([[pelvis_p[0,0]],[pelvis_p[1,0]],[pelvis_p[2,0]],[P_Roll],[P_Pitch],[P_Yaw]])
        self.LX = np.array([[l_foot_p[0,0]],[l_foot_p[1,0]],[l_foot_p[2,0]],[L_Roll],[L_Pitch],[L_Yaw]])
        self.RX = np.array([[r_foot_p[0,0]],[r_foot_p[1,0]],[r_foot_p[2,0]],[R_Roll],[R_Pitch],[R_Yaw]])

        ##////這是啥
        self.L_Body_transfer = np.array([
            [cos(L_Pitch)*cos(L_Yaw), -sin(L_Yaw), 0],
            [cos(L_Pitch)*sin(L_Yaw),  cos(L_Yaw), 0],
            [-sin(L_Pitch),             0,         1]
            ])  
        
        self.R_Body_transfer = np.array([
            [cos(R_Pitch)*cos(R_Yaw), -sin(R_Yaw), 0],
            [cos(R_Pitch)*sin(R_Yaw),  cos(R_Yaw), 0],
            [-sin(R_Pitch),            0,          1]
            ])  
        
        # print("PX",self.PX)
        # print("LX",self.LX)
        # print("RX",self.RX)

        px_in_lf = self.PX - self.LX #骨盆中心相對於左腳
        px_in_rf = self.PX - self.RX #骨盆中心相對於右腳

        return px_in_lf,px_in_rf
 
    def com_position(self,joint_position):
        '''
        回傳(對左腳質點位置，對右腳的，對骨盆的)
        p.s. 不管是哪個模型，原點都在兩隻腳(相距0.2m)中間
        '''
        #get com position
        jp_l = np.reshape(copy.deepcopy(joint_position[0:6,0]),(6,1)) #左腳
        jp_r = np.reshape(copy.deepcopy(joint_position[6:,0]),(6,1))  #右腳
        
        #右腳為支撐腳
        R_jp_r = np.flip(-jp_r,axis=0)
        R_jp_l = jp_l
        R_joint_angle = np.vstack((R_jp_r,R_jp_l))
        pin.centerOfMass(self.bipedal_r_model,self.bipedal_r_data,R_joint_angle)
        com_r_in_pink = np.reshape(self.bipedal_r_data.com[0],(3,1))
        r_foot_in_wf = np.array([[0.0],[-0.1],[0.0]]) ##////其實是pink，原點在兩隻腳正中間
        com_in_rf = com_r_in_pink - r_foot_in_wf

        #左腳為支撐腳
        L_jp_l = np.flip(-jp_l,axis=0)
        L_jp_r = jp_r
        L_joint_angle = np.vstack((L_jp_l,L_jp_r))
        pin.centerOfMass(self.bipedal_l_model,self.bipedal_l_data,L_joint_angle)
        com_l_in_pink = np.reshape(self.bipedal_l_data.com[0],(3,1))
        l_foot_in_wf = np.array([[0.0],[0.1],[0]])
        com_in_lf = com_l_in_pink - l_foot_in_wf

        #floating com ////是從骨盆中建，所以應該可以得知骨盆和質心的位置吧？？
        joint_angle = np.vstack((jp_l,jp_r))
        pin.centerOfMass(self.bipedal_floating_model,self.bipedal_floating_data,joint_angle)
        com_floating_in_pink = np.reshape(self.bipedal_floating_data.com[0],(3,1))
        # print(com_floating_in_pink)

        # print('clf:',com_in_lf)
        # print('crf:',com_in_rf)

        return com_in_lf,com_in_rf,com_floating_in_pink

    def stance_change(self,state,px_in_lf,px_in_rf,stance,contact_t):
        if state == 0:
            stance = 1
            
        elif state == 1:
            stance = 1

        elif state == 2:
            stance = 1
            if self.DS_time <= 10 * self.DDT:
                self.DS_time += self.timer_period
                print("DS",self.DS_time)

        if state == 30:

            #踩到地面才切換支撐腳
            if abs(contact_t-0.5)<=0.005:#(T)
                if self.stance == 1:
                    stance = 0
                    # if self.P_R_wf[2,0] <= 0.01:
                    #     stance = 0
                    # else:
                    #     stance = 1
                elif self.stance == 0:
                    stance = 1
                    # if self.P_L_wf[2,0] <= 0.01:
                    #     stance = 1
                    # else:
                    #     stance = 0
            else:
                 self.stance = stance

        self.stance = stance

        return stance

    def pelvis_in_wf(self):
        '''
        用訂閱到的base對WF的位態,求骨盆對WF的位態
        '''
        P_B_wf = copy.deepcopy(self.P_B_wf)##////base對WF的位置
        O_wfB = copy.deepcopy(self.O_wfB)##////base對WF的旋轉矩陣

        self.P_PV_wf = O_wfB@np.array([[0.0],[0.0],[0.598]]) + P_B_wf
        self.O_wfPV = copy.deepcopy(self.O_wfB)

        # self.PX_publisher.publish(Float64MultiArray(data=self.P_PV_wf))

        return 

    def base_in_wf(self,msg):
        P_base_x = msg.pose.pose.position.x
        P_base_y = msg.pose.pose.position.y
        P_base_z = msg.pose.pose.position.z
        self.P_B_wf = np.array([[P_base_x],[P_base_y],[P_base_z]])

        O_base_x = msg.pose.pose.orientation.x
        O_base_y = msg.pose.pose.orientation.y
        O_base_z = msg.pose.pose.orientation.z
        O_base_w = msg.pose.pose.orientation.w
        base_quaternions = R.from_quat([O_base_x, O_base_y, O_base_z, O_base_w])
        self.O_wfB = base_quaternions.as_matrix()  #注意

    def data_in_wf(self,com_in_pink):
        '''
        就.....一堆轉換
        '''
        #pf_p
        P_PV_pf = copy.deepcopy(self.P_PV_pf)
        P_COM_pf = copy.deepcopy(com_in_pink)

        P_Lhr_pf = copy.deepcopy(self.P_Lhr_pf)
        P_Lhy_pf = copy.deepcopy(self.P_Lhy_pf)
        P_Lhp_pf = copy.deepcopy(self.P_Lhp_pf)
        P_Lkp_pf = copy.deepcopy(self.P_Lkp_pf)
        P_Lap_pf = copy.deepcopy(self.P_Lap_pf)
        P_Lar_pf = copy.deepcopy(self.P_Lar_pf)

        P_L_pf= copy.deepcopy(self.P_L_pf) 

        P_Rhr_pf = copy.deepcopy(self.P_Rhr_pf)
        P_Rhy_pf = copy.deepcopy(self.P_Rhy_pf)
        P_Rhp_pf = copy.deepcopy(self.P_Rhp_pf)
        P_Rkp_pf = copy.deepcopy(self.P_Rkp_pf)
        P_Rap_pf = copy.deepcopy(self.P_Rap_pf)
        P_Rar_pf = copy.deepcopy(self.P_Rar_pf)

        P_R_pf = copy.deepcopy(self.P_R_pf) 

        #pf_o
        O_pfL = copy.deepcopy(self.O_pfL)
        O_pfR = copy.deepcopy(self.O_pfR)
        
        #PV_o ////直走所以是單位矩陣，沒有旋轉
        O_PVpf = np.identity(3)
        #wf_p
        P_PV_wf = copy.deepcopy(self.P_PV_wf) #ros-3d
        O_wfPV = copy.deepcopy(self.O_wfPV) #ros-3d
        P_COM_wf = O_wfPV@O_PVpf@(P_COM_pf - P_PV_pf) + P_PV_wf

        P_Lhr_wf = O_wfPV@O_PVpf@(P_Lhr_pf - P_PV_pf) + P_PV_wf
        P_Lhy_wf = O_wfPV@O_PVpf@(P_Lhy_pf - P_PV_pf) + P_PV_wf
        P_Lhp_wf = O_wfPV@O_PVpf@(P_Lhp_pf - P_PV_pf) + P_PV_wf
        P_Lkp_wf = O_wfPV@O_PVpf@(P_Lkp_pf - P_PV_pf) + P_PV_wf
        P_Lap_wf = O_wfPV@O_PVpf@(P_Lap_pf - P_PV_pf) + P_PV_wf
        P_Lar_wf = O_wfPV@O_PVpf@(P_Lar_pf - P_PV_pf) + P_PV_wf
    
        P_L_wf = O_wfPV@O_PVpf@(P_L_pf - P_PV_pf) + P_PV_wf

        P_Rhr_wf = O_wfPV@O_PVpf@(P_Rhr_pf - P_PV_pf) + P_PV_wf
        P_Rhy_wf = O_wfPV@O_PVpf@(P_Rhy_pf - P_PV_pf) + P_PV_wf
        P_Rhp_wf = O_wfPV@O_PVpf@(P_Rhp_pf - P_PV_pf) + P_PV_wf
        P_Rkp_wf = O_wfPV@O_PVpf@(P_Rkp_pf - P_PV_pf) + P_PV_wf
        P_Rap_wf = O_wfPV@O_PVpf@(P_Rap_pf - P_PV_pf) + P_PV_wf
        P_Rar_wf = O_wfPV@O_PVpf@(P_Rar_pf - P_PV_pf) + P_PV_wf

        P_R_wf = O_wfPV@O_PVpf@(P_R_pf - P_PV_pf) + P_PV_wf

        #wf_o
        O_wfR = O_wfPV@O_PVpf@O_pfR
        O_wfL = O_wfPV@O_PVpf@O_pfL

        #assign data for global use
        #position in wf
        self.P_PV_wf = copy.deepcopy(P_PV_wf)
        self.P_COM_wf = copy.deepcopy(P_COM_wf)

        self.P_Lhr_wf = copy.deepcopy(P_Lhr_wf)
        self.P_Lhy_wf = copy.deepcopy(P_Lhy_wf)
        self.P_Lhp_wf = copy.deepcopy(P_Lhp_wf)
        self.P_Lkp_wf = copy.deepcopy(P_Lkp_wf)
        self.P_Lap_wf = copy.deepcopy(P_Lap_wf)
        self.P_Lar_wf = copy.deepcopy(P_Lar_wf)

        self.P_L_wf = copy.deepcopy(P_L_wf)

        self.P_Rhr_wf = copy.deepcopy(P_Rhr_wf)
        self.P_Rhy_wf = copy.deepcopy(P_Rhy_wf)
        self.P_Rhp_wf = copy.deepcopy(P_Rhp_wf)
        self.P_Rkp_wf = copy.deepcopy(P_Rkp_wf)
        self.P_Rap_wf = copy.deepcopy(P_Rap_wf)
        self.P_Rar_wf = copy.deepcopy(P_Rar_wf)

        self.P_R_wf = copy.deepcopy(P_R_wf)
        #orientation in wf
        self.O_wfPV = copy.deepcopy(O_wfPV)
        self.O_wfL = copy.deepcopy(O_wfL)
        self.O_wfR = copy.deepcopy(O_wfR)

        # self.PX_publisher.publish(Float64MultiArray(data=P_PV_wf))
        # self.COM_publisher.publish(Float64MultiArray(data=P_COM_wf))
        # self.LX_publisher.publish(Float64MultiArray(data=P_L_wf))
        # self.RX_publisher.publish(Float64MultiArray(data=P_R_wf))

        return 

    def rotation_matrix(self,joint_position):
        jp = copy.deepcopy(joint_position)
        
        # 各關節角度
        Theta1 = jp[0,0] #L_Hip_Roll
        Theta2 = jp[1,0]
        Theta3 = jp[2,0]
        Theta4 = jp[3,0]
        Theta5 = jp[4,0]
        Theta6 = jp[5,0] #L_Ankle_Roll

        Theta7 = jp[6,0] #R_Hip_Roll
        Theta8 = jp[7,0]
        Theta9 = jp[8,0]
        Theta10 = jp[9,0]
        Theta11 = jp[10,0]
        Theta12 = jp[11,0] #R_Ankle_Roll

        #calculate rotation matrix
        self.L_R01 = self.xyz_rotation('x',Theta1) #L_Hip_roll
        self.L_R12 = self.xyz_rotation('z',Theta2)
        self.L_R23 = self.xyz_rotation('y',Theta3)
        self.L_R34 = self.xyz_rotation('y',Theta4)
        self.L_R45 = self.xyz_rotation('y',Theta5)
        self.L_R56 = self.xyz_rotation('x',Theta6) #L_Ankle_roll

        self.R_R01 = self.xyz_rotation('x',Theta7) #R_Hip_roll
        self.R_R12 = self.xyz_rotation('z',Theta8)
        self.R_R23 = self.xyz_rotation('y',Theta9)
        self.R_R34 = self.xyz_rotation('y',Theta10)
        self.R_R45 = self.xyz_rotation('y',Theta11)
        self.R_R56 = self.xyz_rotation('x',Theta12) #R_Ankle_roll

    def relative_axis(self):
        '''
        不知道是幹麻的
        '''
        #骨盆姿態(要確認！)
        self.RP = np.array([[1,0,0],[0,1,0],[0,0,1]])
        # self.RP = copy.deepcopy(self.O_wfPV)

        self.AL1 = self.RP@(np.array([[1],[0],[0]])) #L_Hip_roll
        self.AL2 = self.RP@self.L_R01@(np.array([[0],[0],[1]])) 
        self.AL3 = self.RP@self.L_R01@self.L_R12@(np.array([[0],[1],[0]])) 
        self.AL4 = self.RP@self.L_R01@self.L_R12@self.L_R23@(np.array([[0],[1],[0]]))
        self.AL5 = self.RP@self.L_R01@self.L_R12@self.L_R23@self.L_R34@(np.array([[0],[1],[0]])) 
        self.AL6 = self.RP@self.L_R01@self.L_R12@self.L_R23@self.L_R34@self.L_R45@(np.array([[1],[0],[0]])) #L_Ankle_Roll

        self.AR1 = self.RP@(np.array([[1],[0],[0]])) #R_Hip_roll
        self.AR2 = self.RP@self.R_R01@(np.array([[0],[0],[1]])) 
        self.AR3 = self.RP@self.R_R01@self.R_R12@(np.array([[0],[1],[0]])) 
        self.AR4 = self.RP@self.R_R01@self.R_R12@self.R_R23@(np.array([[0],[1],[0]]))
        self.AR5 = self.RP@self.R_R01@self.R_R12@self.R_R23@self.R_R34@(np.array([[0],[1],[0]])) 
        self.AR6 = self.RP@self.R_R01@self.R_R12@self.R_R23@self.R_R34@self.R_R45@(np.array([[1],[0],[0]])) #R_Ankle_Roll

    def get_initial_data(self,stance):
        P_L_wf = copy.deepcopy(self.P_L_wf)
        P_R_wf = copy.deepcopy(self.P_R_wf)
        #怎麼順利地拿?
        #直接拿切換後的支撐腳所估測出的狀態當成初始狀態不合理 因為該估測狀態所用的扭矩來自該腳仍是擺動腳時所得到的
        #所以初始狀態選擇拿量測值(我現在的想法)

        #藉由支撐狀態切換
        if stance == 1: #(左單支撐)
            #支撐frame
            P_cf_wf = P_L_wf
            O_wfcf = np.array([[1,0,0],[0,1,0],[0,0,1]])
            #初始狀態
            X0 = copy.deepcopy(self.mea_x_L)
            Y0 = copy.deepcopy(self.mea_y_L)
            #擺動腳前一刻狀態
            O_cfwf = np.transpose(O_wfcf)
            Psw2com_X_0 = O_cfwf@(self.P_COM_wf - self.P_R_wf)
            Psw2com_0 = np.array([[Psw2com_X_0[0,0]],[Psw2com_X_0[1,0]]])
        else:#(右單支撐)
            #支撐frame
            P_cf_wf = P_R_wf
            O_wfcf = np.array([[1,0,0],[0,1,0],[0,0,1]])
            #初始狀態
            X0 = copy.deepcopy(self.mea_x_R)
            Y0 = copy.deepcopy(self.mea_y_R)
            #擺動腳前一刻狀態
            O_cfwf = np.transpose(O_wfcf)
            Psw2com_X_0 = O_cfwf@(self.P_COM_wf - self.P_L_wf)
            Psw2com_0 = np.array([[Psw2com_X_0[0,0]],[Psw2com_X_0[1,0]]])
        
        return P_cf_wf,X0,Y0,Psw2com_0

    def online_planning(self,stance,contact_t,P_cf_wf,X0,Y0,P_Psw2com_0):

        t = copy.deepcopy(contact_t)#該支撐狀態運行時間
        P_cf_wf = copy.deepcopy(P_cf_wf) #contact frame 在 wf 上的位置(xyz)
        O_wfcf = np.array([[1,0,0],[0,1,0],[0,0,1]]) #contact frame 在 wf 上的姿態
        X0 = copy.deepcopy(X0) #切換至該支撐下，初始狀態(xc(0)、ly(0))
        Y0 = copy.deepcopy(Y0) #切換至該支撐下，初始狀態(yc(0)、lx(0))
        Psw2com_x_0 = copy.deepcopy(P_Psw2com_0[0,0])
        Psw2com_y_0 = copy.deepcopy(P_Psw2com_0[1,0])
        cosh = math.cosh
        sinh = math.sinh
        cos = math.cos
        sin = math.sin
        pi = math.pi

        #理想機器人狀態
        m = 9    #機器人下肢總重
        H = 0.45 #理想質心高度
        W = 0.2 #兩腳底間距
        g = 9.81 #重力
        l = math.sqrt(g/H)


        #會用到的參數
        T = 0.5 #支撐間格時長
        Vx_des_2T = 0.15 #下兩步踩踏時刻x方向理想速度
        Ly_des_2T = m*Vx_des_2T*H #下兩步踩踏時刻相對於接觸點的理想y方向角動量
        #下兩步踩踏時刻相對於接觸點的理想x方向角動量
        Lx_des_2T_1 = (0.5*m*H*W)*(l*sinh(l*T))/(1+cosh(l*T)) #當下一次支撐腳是左腳(現在是右單支)
        Lx_des_2T_2 = -(0.5*m*H*W)*(l*sinh(l*T))/(1+cosh(l*T)) #當下一次支撐腳是右腳(現在是左單支)
        #踏步高度
        zCL = 0.02
        
        # print("t",t)
        
        #理想ALIP模型動態
        ALIP_x = np.array([[cosh(l*t),(sinh(l*t)/(m*H*l))],[m*H*l*sinh(l*t),cosh(l*t)]])
        ALIP_y = np.array([[cosh(l*t),-(sinh(l*t)/(m*H*l))],[-m*H*l*sinh(l*t),cosh(l*t)]])
        
        # print(" ALIP_x", ALIP_x)
        # print("X0",X0)

        #質心參考軌跡
        #理想上，質心相對接觸點的frame的位置及角動量(用ALIP求)，這會被拿去當成支撐腳ALIP的參考命令
        Xx_cf = np.reshape(ALIP_x@X0,(2,1))#(xc、ly)
        Xy_cf = np.reshape(ALIP_y@Y0,(2,1))#(yc、lx)

        #debug
        # print("Xx:",Xx_cf)
        
        #理想上，質心相對接觸點的位置(x,y,z)
        Com_x_cf = copy.deepcopy(Xx_cf[0,0])
        Com_y_cf = copy.deepcopy(Xy_cf[0,0])
        Com_z_cf = copy.deepcopy(H)
        #轉成大地座標下的軌跡
        Com_ref_wf = O_wfcf@np.array([[Com_x_cf],[Com_y_cf],[Com_z_cf]]) + P_cf_wf
       
        #支撐腳參考軌跡
        Support_ref_wf = P_cf_wf#延續

        #擺動腳參考軌跡
        #更新下一次踩踏瞬間時理想角動量數值
        Ly_T = m*H*l*sinh(l*T)*X0[0,0] + cosh(l*T)*X0[1,0]
        Lx_T = -m*H*l*sinh(l*T)*Y0[0,0] + cosh(l*T)*Y0[1,0]
        #根據下一次支撐腳切換lx
        if stance == 1:
            Lx_des_2T = Lx_des_2T_2
        else:
            Lx_des_2T = Lx_des_2T_1
        #理想上，下一步擺動腳踩踏點(相對於下一步踩踏時刻下的質心位置)
        Psw2com_x_T = (Ly_des_2T - cosh(l*T)*Ly_T)/(m*H*l*sinh(l*T))
        Psw2com_y_T = (Lx_des_2T - cosh(l*T)*Lx_T)/-(m*H*l*sinh(l*T))
        #理想上，擺動腳相對接觸點的位置(x,y,z)
        pv = t/T #變數 用於連接擺動腳軌跡
        Sw_x_cf = Com_x_cf - (0.5*((1+cos(pi*pv))*Psw2com_x_0 + (1-cos(pi*pv))*Psw2com_x_T))
        Sw_y_cf = Com_y_cf - (0.5*((1+cos(pi*pv))*Psw2com_y_0 + (1-cos(pi*pv))*Psw2com_y_T))
        Sw_z_cf = Com_z_cf - (4*zCL*(pv-0.5)**2 + (H-zCL))
        #轉成大地座標下的軌跡
        Swing_ref_wf = O_wfcf@np.array([[Sw_x_cf],[Sw_y_cf],[Sw_z_cf]]) + P_cf_wf

        #根據支撐狀態分配支撐腳軌跡、擺動腳軌跡、ALIP參考軌跡(質心、角動量)
        if stance == 1:
            L_ref_wf = Support_ref_wf
            R_ref_wf = Swing_ref_wf
            self.ref_x_L = copy.deepcopy(np.array([[Xx_cf[0,0]],[Xx_cf[1,0]]]))
            self.ref_y_L = copy.deepcopy(np.array([[Xy_cf[0,0]],[Xy_cf[1,0]]]))
        else:
            L_ref_wf = Swing_ref_wf
            R_ref_wf = Support_ref_wf
            self.ref_x_R = np.array([[Xx_cf[0,0]],[Xx_cf[1,0]]])
            self.ref_y_R = np.array([[Xy_cf[0,0]],[Xy_cf[1,0]]])
        
        return Com_ref_wf,L_ref_wf,R_ref_wf
  
    def ref_alip(self,stance,px_in_lf,px_in_rf,com_in_lf,com_in_rf,Com_ref_wf,L_ref_wf,R_ref_wf):
        #ALIP
        L_X_ref = L_ref_wf[0,0]
        L_Y_ref = L_ref_wf[1,0]
        L_Z_ref = L_ref_wf[2,0]

        R_X_ref = R_ref_wf[0,0]
        R_Y_ref = R_ref_wf[1,0]
        R_Z_ref = R_ref_wf[2,0]

        if stance == 1:
            #取骨盆跟質心在位置上的差異(in_wf)
            P_px_lf = np.reshape(copy.deepcopy(px_in_lf[0:3,0]),(3,1))
            com2px_in_lf = P_px_lf - com_in_lf
            O_wfL = np.array([[1,0,0],[0,1,0],[0,0,1]]) #直走&理想
            com2px_in_wf = O_wfL@(com2px_in_lf)
            #將質心軌跡更改成骨盆軌跡(in_wf)
            P_X_ref = Com_ref_wf[0,0] + (com2px_in_wf[0,0])
            P_Y_ref = Com_ref_wf[1,0] + (com2px_in_wf[1,0])
            P_Z_ref = 0.55

        elif stance == 0:
            #取骨盆跟質心在位置上的差異(in_wf)
            P_px_rf = np.reshape(copy.deepcopy(px_in_rf[0:3,0]),(3,1))
            com2px_in_rf = P_px_rf - com_in_rf
            O_wfR = np.array([[1,0,0],[0,1,0],[0,0,1]]) #直走&理想
            com2px_in_wf = O_wfR@(com2px_in_rf)       
            #將質心軌跡更改成骨盆軌跡(in_wf)
            P_X_ref = Com_ref_wf[0,0] + (com2px_in_wf[0,0])
            P_Y_ref = Com_ref_wf[1,0] + (com2px_in_wf[1,0])
            P_Z_ref = 0.55
        
        #直走下預設為0
        #骨盆姿態
        P_Roll_ref = 0.0
        P_Pitch_ref = 0.0
        P_Yaw_ref = 0.0
        #左腳腳底姿態
        L_Roll_ref = 0.0
        L_Pitch_ref = 0.0
        L_Yaw_ref = 0.0
        #右腳腳底姿態        
        R_Roll_ref = 0.0
        R_Pitch_ref = 0.0
        R_Yaw_ref = 0.0

        self.PX_ref = np.array([[P_X_ref],[P_Y_ref],[P_Z_ref],[P_Roll_ref],[P_Pitch_ref],[P_Yaw_ref]])
        self.LX_ref = np.array([[L_X_ref],[L_Y_ref],[L_Z_ref],[L_Roll_ref],[L_Pitch_ref],[L_Yaw_ref]])
        self.RX_ref = np.array([[R_X_ref],[R_Y_ref],[R_Z_ref],[R_Roll_ref],[R_Pitch_ref],[R_Yaw_ref]]) 


    
    def left_leg_jacobian(self):
        pelvis = copy.deepcopy(self.P_PV_pf)
        l_hip_roll = copy.deepcopy(self.P_Lhr_pf)
        l_hip_yaw = copy.deepcopy(self.P_Lhy_pf)
        l_hip_pitch = copy.deepcopy(self.P_Lhp_pf)
        l_knee_pitch = copy.deepcopy(self.P_Lkp_pf)
        l_ankle_pitch = copy.deepcopy(self.P_Lap_pf)
        l_ankle_roll = copy.deepcopy(self.P_Lar_pf)
        l_foot = copy.deepcopy(self.P_L_pf)

        JL1 = np.cross(self.AL1,(l_foot-l_hip_roll),axis=0)
        JL2 = np.cross(self.AL2,(l_foot-l_hip_yaw),axis=0)
        JL3 = np.cross(self.AL3,(l_foot-l_hip_pitch),axis=0)
        JL4 = np.cross(self.AL4,(l_foot-l_knee_pitch),axis=0)
        JL5 = np.cross(self.AL5,(l_foot-l_ankle_pitch),axis=0)
        JL6 = np.cross(self.AL6,(l_foot-l_ankle_roll),axis=0)

        JLL_upper = np.hstack((JL1, JL2,JL3,JL4,JL5,JL6))
        JLL_lower = np.hstack((self.AL1,self.AL2,self.AL3,self.AL4,self.AL5,self.AL6))    
        self.JLL = np.vstack((JLL_upper,JLL_lower))  
        # print(self.JLL)

        #排除支撐腳腳踝對末端速度的影響
        self.JL_sp44 = np.reshape(self.JLL[2:,0:4],(4,4))  
        self.JL_sp42 = np.reshape(self.JLL[2:,4:],(4,2))
        #排除擺動.腳踝對末端速度的影響
        JL_sw34 = np.reshape(self.JLL[0:3,0:4],(3,4)) 
        JL_sw14 = np.reshape(self.JLL[5,0:4],(1,4)) 
        self.JL_sw44 = np.vstack((JL_sw34,JL_sw14))

        JL_sw32 = np.reshape(self.JLL[0:3,4:],(3,2)) 
        JL_sw12 = np.reshape(self.JLL[5,4:],(1,2)) 
        self.JL_sw42 = np.vstack((JL_sw32,JL_sw12))

        return self.JLL

    def left_leg_jacobian_wf(self):
        pelvis = copy.deepcopy(self.P_PV_wf)
        l_hip_roll = copy.deepcopy(self.P_Lhr_wf)
        l_hip_yaw = copy.deepcopy(self.P_Lhy_wf)
        l_hip_pitch = copy.deepcopy(self.P_Lhp_wf)
        l_knee_pitch = copy.deepcopy(self.P_Lkp_wf)
        l_ankle_pitch = copy.deepcopy(self.P_Lap_wf)
        l_ankle_roll = copy.deepcopy(self.P_Lar_wf)
        l_foot = copy.deepcopy(self.P_L_wf)

        JL1 = np.cross(self.AL1,(l_foot-l_hip_roll),axis=0)
        JL2 = np.cross(self.AL2,(l_foot-l_hip_yaw),axis=0)
        JL3 = np.cross(self.AL3,(l_foot-l_hip_pitch),axis=0)
        JL4 = np.cross(self.AL4,(l_foot-l_knee_pitch),axis=0)
        JL5 = np.cross(self.AL5,(l_foot-l_ankle_pitch),axis=0)
        JL6 = np.cross(self.AL6,(l_foot-l_ankle_roll),axis=0)

        JLL_upper = np.hstack((JL1, JL2,JL3,JL4,JL5,JL6))
        JLL_lower = np.hstack((self.AL1,self.AL2,self.AL3,self.AL4,self.AL5,self.AL6))    
        self.JLL = np.vstack((JLL_upper,JLL_lower))  
        # print(self.JLL)

        #排除支撐腳腳踝對末端速度的影響
        self.JL_sp44 = np.reshape(self.JLL[2:,0:4],(4,4))  
        self.JL_sp42 = np.reshape(self.JLL[2:,4:],(4,2))

        return self.JLL

    def right_leg_jacobian(self):
        pelvis = copy.deepcopy(self.P_PV_pf)
        r_hip_roll = copy.deepcopy(self.P_Rhr_pf)
        r_hip_yaw = copy.deepcopy(self.P_Rhy_pf)
        r_hip_pitch = copy.deepcopy(self.P_Rhp_pf)
        r_knee_pitch = copy.deepcopy(self.P_Rkp_pf)
        r_ankle_pitch = copy.deepcopy(self.P_Rap_pf)
        r_ankle_roll = copy.deepcopy(self.P_Rar_pf)
        r_foot = copy.deepcopy(self.P_R_pf)

        JR1 = np.cross(self.AR1,(r_foot-r_hip_roll),axis=0)
        JR2 = np.cross(self.AR2,(r_foot-r_hip_yaw),axis=0)
        JR3 = np.cross(self.AR3,(r_foot-r_hip_pitch),axis=0)
        JR4 = np.cross(self.AR4,(r_foot-r_knee_pitch),axis=0)
        JR5 = np.cross(self.AR5,(r_foot-r_ankle_pitch),axis=0)
        JR6 = np.cross(self.AR6,(r_foot-r_ankle_roll),axis=0)

        JRR_upper = np.hstack((JR1,JR2,JR3,JR4,JR5,JR6))
        JRR_lower = np.hstack((self.AR1,self.AR2,self.AR3,self.AR4,self.AR5,self.AR6))    
        self.JRR = np.vstack((JRR_upper,JRR_lower))  
        # print(self.JRR)

        #排除支撐腳腳踝對末端速度的影響
        self.JR_sp44 = np.reshape(self.JRR[2:,0:4],(4,4))  
        self.JR_sp42 = np.reshape(self.JRR[2:,4:],(4,2))
        #排除擺動.腳踝對末端速度的影響
        JR_sw34 = np.reshape(self.JRR[0:3,0:4],(3,4)) 
        JR_sw14 = np.reshape(self.JRR[5,0:4],(1,4)) 
        self.JR_sw44 = np.vstack((JR_sw34,JR_sw14))

        JR_sw32 = np.reshape(self.JRR[0:3,4:],(3,2)) 
        JR_sw12 = np.reshape(self.JRR[5,4:],(1,2)) 
        self.JR_sw42 = np.vstack((JR_sw32,JR_sw12))

        return self.JRR

    def right_leg_jacobian_wf(self):
        pelvis = copy.deepcopy(self.P_PV_wf)
        r_hip_roll = copy.deepcopy(self.P_Rhr_wf)
        r_hip_yaw = copy.deepcopy(self.P_Rhy_wf)
        r_hip_pitch = copy.deepcopy(self.P_Rhp_wf)
        r_knee_pitch = copy.deepcopy(self.P_Rkp_wf)
        r_ankle_pitch = copy.deepcopy(self.P_Rap_wf)
        r_ankle_roll = copy.deepcopy(self.P_Rar_wf)
        r_foot = copy.deepcopy(self.P_R_wf)

        JR1 = np.cross(self.AR1,(r_foot-r_hip_roll),axis=0)
        JR2 = np.cross(self.AR2,(r_foot-r_hip_yaw),axis=0)
        JR3 = np.cross(self.AR3,(r_foot-r_hip_pitch),axis=0)
        JR4 = np.cross(self.AR4,(r_foot-r_knee_pitch),axis=0)
        JR5 = np.cross(self.AR5,(r_foot-r_ankle_pitch),axis=0)
        JR6 = np.cross(self.AR6,(r_foot-r_ankle_roll),axis=0)

        JRR_upper = np.hstack((JR1,JR2,JR3,JR4,JR5,JR6))
        JRR_lower = np.hstack((self.AR1,self.AR2,self.AR3,self.AR4,self.AR5,self.AR6))    
        self.JRR = np.vstack((JRR_upper,JRR_lower))  
        # print(self.JRR)

        #排除支撐腳腳踝對末端速度的影響
        self.JR_sp44 = np.reshape(self.JRR[2:,0:4],(4,4))  
        self.JR_sp42 = np.reshape(self.JRR[2:,4:],(4,2))
        return self.JRR
   
    def gravity_compemsate(self,joint_position,stance_type,px_in_lf,px_in_rf,l_contact,r_contact,state):

        jp_l = np.reshape(copy.deepcopy(joint_position[0:6,0]),(6,1)) #左腳
        jp_r = np.reshape(copy.deepcopy(joint_position[6:,0]),(6,1))  #右腳
        stance = copy.deepcopy((stance_type))
        
        #DS_gravity
        jp_L_DS = np.flip(-jp_l,axis=0)
        jv_L_DS = np.zeros((6,1))
        c_L_DS = np.zeros((6,1))
        L_DS_gravity = np.reshape(-pin.rnea(self.stance_l_model, self.stance_l_data, jp_L_DS,jv_L_DS,(c_L_DS)),(6,1))  
        L_DS_gravity = np.flip(L_DS_gravity,axis=0)

        jp_R_DS = np.flip(-jp_r,axis=0)
        jv_R_DS = np.zeros((6,1))
        c_R_DS = np.zeros((6,1))
        R_DS_gravity = np.reshape(-pin.rnea(self.stance_r_model, self.stance_r_data, jp_R_DS,jv_R_DS,(c_R_DS)),(6,1))  
        R_DS_gravity = np.flip(R_DS_gravity,axis=0)
        DS_gravity = np.vstack((L_DS_gravity, R_DS_gravity))

        #RSS_gravity
        jp_R_RSS = np.flip(-jp_r,axis=0)
        jp_RSS = np.vstack((jp_R_RSS,jp_l))
        jv_RSS = np.zeros((12,1))
        c_RSS = np.zeros((12,1))
        Leg_RSS_gravity = np.reshape(pin.rnea(self.bipedal_r_model, self.bipedal_r_data, jp_RSS,jv_RSS,(c_RSS)),(12,1))  

        L_RSS_gravity = np.reshape(Leg_RSS_gravity[6:,0],(6,1))
        R_RSS_gravity = np.reshape(-Leg_RSS_gravity[0:6,0],(6,1)) #加負號(相對關係)
        R_RSS_gravity = np.flip(R_RSS_gravity,axis=0)
        RSS_gravity = np.vstack((L_RSS_gravity, R_RSS_gravity))

        #LSS_gravity
        jp_L_LSS = np.flip(-jp_l,axis=0)
        jp_LSS = np.vstack((jp_L_LSS,jp_r))
        jv_LSS = np.zeros((12,1))
        c_LSS = np.zeros((12,1))
        Leg_LSS_gravity = np.reshape(pin.rnea(self.bipedal_l_model, self.bipedal_l_data, jp_LSS,jv_LSS,(c_LSS)),(12,1))  

        L_LSS_gravity = np.reshape(-Leg_LSS_gravity[0:6,0],(6,1)) #加負號(相對關係)
        L_LSS_gravity = np.flip(L_LSS_gravity,axis=0)
        R_LSS_gravity = np.reshape(Leg_LSS_gravity[6:,0],(6,1))
        LSS_gravity = np.vstack((L_LSS_gravity, R_LSS_gravity))

        if stance == 2:
            if r_contact == 1:
                kr = np.array([[1.2],[1.2],[1.2],[1.2],[1.2],[1.2]])
            else:
                kr = np.array([[1],[1],[1],[1],[1],[1]])
            if l_contact == 1:
                kl = np.array([[1.2],[1.2],[1.2],[1.2],[1.2],[1.2]])
            else:
                kl = np.array([[1],[1],[1],[1],[1],[1]])

            if abs(px_in_lf[1,0]) < abs(px_in_rf[1,0]):
                Leg_gravity = (abs(px_in_lf[1,0])/0.1)*DS_gravity + ((0.1-abs(px_in_lf[1,0]))/0.1)*LSS_gravity
            
            elif abs(px_in_rf[1,0])< abs(px_in_lf[1,0]):
                Leg_gravity = (abs(px_in_rf[1,0])/0.1)*DS_gravity + ((0.1-abs(px_in_rf[1,0]))/0.1)*RSS_gravity
            
            else:
                Leg_gravity = DS_gravity

            # if abs(px_in_rf[1,0])<=0.05 and r_contact ==1:
            #     Leg_gravity = (abs(px_in_rf[1,0])/0.05)*DS_gravity + ((0.05-abs(px_in_rf[1,0]))/0.05)*RSS_gravity
            
            # elif abs(px_in_lf[1,0])<=0.05 and l_contact ==1:
            #     Leg_gravity = (abs(px_in_lf[1,0])/0.05)*DS_gravity + ((0.05-abs(px_in_lf[1,0]))/0.05)*LSS_gravity
            
            # else:
            #     Leg_gravity = DS_gravity
        
        elif stance == 0:
            if r_contact == 1:
                kr = np.array([[1.2],[1.2],[1.2],[1.2],[1.2],[1.2]])
            else:
                kr = np.array([[1],[1],[1],[1],[1],[1]])
            if l_contact == 1:
                kl = np.array([[1.2],[1.2],[1.2],[1.2],[1.2],[1.2]])
            else:
                kl = np.array([[1],[1],[1],[1],[1],[1]])

            if abs(px_in_lf[1,0]) < abs(px_in_rf[1,0]):
                Leg_gravity = (abs(px_in_lf[1,0])/0.1)*DS_gravity + ((0.1-abs(px_in_lf[1,0]))/0.1)*LSS_gravity
            
            elif abs(px_in_rf[1,0])< abs(px_in_lf[1,0]):
                Leg_gravity = (abs(px_in_rf[1,0])/0.1)*DS_gravity + ((0.1-abs(px_in_rf[1,0]))/0.1)*RSS_gravity
            
            else:
                Leg_gravity = DS_gravity
            # if r_contact == 1:
            #     kr = np.array([[1.2],[1.2],[1.2],[1.2],[1.5],[1.5]])
            # else:
            #     kr = np.array([[1.2],[1.2],[1.2],[1.2],[1.2],[1.2]])
            # kl = np.array([[1],[1],[1],[0.8],[0.8],[0.8]])
            # Leg_gravity = (px_in_rf[1,0]/0.1)*DS_gravity + ((0.1-px_in_rf[1,0])/0.1)*RSS_gravity
                
            # # if l_contact ==1:
            # #     Leg_gravity = (px_in_rf[1,0]/0.1)*DS_gravity + ((0.1-px_in_rf[1,0])/0.1)*RSS_gravity
            # # else:
            # #     Leg_gravity = RSS_gravity
     
        elif stance == 1:
            if r_contact == 1:
                kr = np.array([[1.2],[1.2],[1.2],[1.2],[1.2],[1.2]])
            else:
                kr = np.array([[1],[1],[1],[1],[1],[1]])
            if l_contact == 1:
                kl = np.array([[1.2],[1.2],[1.2],[1.2],[1.2],[1.2]])
            else:
                kl = np.array([[1],[1],[1],[1],[1],[1]])

            if abs(px_in_lf[1,0]) < abs(px_in_rf[1,0]):
                Leg_gravity = (abs(px_in_lf[1,0])/0.1)*DS_gravity + ((0.1-abs(px_in_lf[1,0]))/0.1)*LSS_gravity
            
            elif abs(px_in_rf[1,0])< abs(px_in_lf[1,0]):
                Leg_gravity = (abs(px_in_rf[1,0])/0.1)*DS_gravity + ((0.1-abs(px_in_rf[1,0]))/0.1)*RSS_gravity
            
            else:
                Leg_gravity = DS_gravity

            # if l_contact == 1:
            #     kl = np.array([[1.2],[1.2],[1.2],[1.2],[1.5],[1.5]])
            # else:
            #     kl = np.array([[1.2],[1.2],[1.2],[1.2],[1.2],[1.2]])
            # Leg_gravity = (-px_in_lf[1,0]/0.1)*DS_gravity + ((0.1+px_in_lf[1,0])/0.1)*LSS_gravity
                
            # # if r_contact ==1:
            # #     Leg_gravity = (-px_in_lf[1,0]/0.1)*DS_gravity + ((0.1+px_in_lf[1,0])/0.1)*LSS_gravity
            # # else:
            # #     Leg_gravity = LSS_gravity


        if state == 1:
            kr = np.array([[0.5],[0.5],[0.5],[0.5],[0.5],[0.5]])
            kl = np.array([[0.5],[0.5],[0.5],[0.5],[0.5],[0.5]])
        
        if state == 2:
            kr = np.array([[0.5],[0.5],[0.5],[0.5],[0.5],[0.5]])
            kl = np.array([[0.5],[0.5],[0.5],[0.5],[0.5],[0.5]])

        l_leg_gravity = np.reshape(Leg_gravity[0:6,0],(6,1))
        r_leg_gravity = np.reshape(Leg_gravity[6:,0],(6,1))

        self.l_gravity_publisher.publish(Float64MultiArray(data=l_leg_gravity))
        self.r_gravity_publisher.publish(Float64MultiArray(data=r_leg_gravity))
        
        return l_leg_gravity,r_leg_gravity,kl,kr
    
    def gravity_ALIP(self,joint_position,stance_type,px_in_lf,px_in_rf,l_contact,r_contact):
        jp_l = np.reshape(copy.deepcopy(joint_position[0:6,0]),(6,1)) #左腳
        jp_r = np.reshape(copy.deepcopy(joint_position[6:,0]),(6,1))  #右腳
        stance = copy.deepcopy((stance_type))

        #DS_gravity
        jp_L_DS = np.flip(-jp_l,axis=0)
        jv_L_DS = np.zeros((6,1))
        c_L_DS = np.zeros((6,1))
        L_DS_gravity = np.reshape(-pin.rnea(self.stance_l_model, self.stance_l_data, jp_L_DS,jv_L_DS,(c_L_DS)),(6,1))  
        L_DS_gravity = np.flip(L_DS_gravity,axis=0)

        jp_R_DS = np.flip(-jp_r,axis=0)
        jv_R_DS = np.zeros((6,1))
        c_R_DS = np.zeros((6,1))
        R_DS_gravity = np.reshape(-pin.rnea(self.stance_r_model, self.stance_r_data, jp_R_DS,jv_R_DS,(c_R_DS)),(6,1))  
        R_DS_gravity = np.flip(R_DS_gravity,axis=0)
        DS_gravity = np.vstack((L_DS_gravity, R_DS_gravity))

        #RSS_gravity
        jp_R_RSS = np.flip(-jp_r,axis=0)
        jp_RSS = np.vstack((jp_R_RSS,jp_l))
        jv_RSS = np.zeros((12,1))
        c_RSS = np.zeros((12,1))
        Leg_RSS_gravity = np.reshape(pin.rnea(self.bipedal_r_model, self.bipedal_r_data, jp_RSS,jv_RSS,(c_RSS)),(12,1))  

        L_RSS_gravity = np.reshape(Leg_RSS_gravity[6:,0],(6,1))
        R_RSS_gravity = np.reshape(-Leg_RSS_gravity[0:6,0],(6,1)) #加負號(相對關係)
        R_RSS_gravity = np.flip(R_RSS_gravity,axis=0)
        RSS_gravity = np.vstack((L_RSS_gravity, R_RSS_gravity))

        #LSS_gravity
        jp_L_LSS = np.flip(-jp_l,axis=0)
        jp_LSS = np.vstack((jp_L_LSS,jp_r))
        jv_LSS = np.zeros((12,1))
        c_LSS = np.zeros((12,1))
        Leg_LSS_gravity = np.reshape(pin.rnea(self.bipedal_l_model, self.bipedal_l_data, jp_LSS,jv_LSS,(c_LSS)),(12,1))  

        L_LSS_gravity = np.reshape(-Leg_LSS_gravity[0:6,0],(6,1)) #加負號(相對關係)
        L_LSS_gravity = np.flip(L_LSS_gravity,axis=0)
        R_LSS_gravity = np.reshape(Leg_LSS_gravity[6:,0],(6,1))
        LSS_gravity = np.vstack((L_LSS_gravity, R_LSS_gravity))

        # if r_contact == 1:
        #     if self.stance_past == 1:
        #         kr = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
        #         kl = np.array([[1.2],[1.2],[1.2],[1.5],[1.5],[1.5]])
        #         # zero_gravity = np.zeros((6,1))
        #         # Leg_gravity = np.vstack((L_LSS_gravity, zero_gravity))
        #         Leg_gravity = 0.15*DS_gravity+0.85*RSS_gravity
        #     else:
        #         kr = np.array([[1.2],[1.2],[1.2],[1.5],[1.5],[1.5]])
        #         kl = np.array([[1],[1],[1],[1],[1],[1]])
        #         Leg_gravity = 0.15*DS_gravity+0.85*RSS_gravity
        
        # elif l_contact == 1:
        #     if self.stance_past == 0:
        #         kl = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
        #         kr = np.array([[1.2],[1.2],[1.2],[1.5],[1.5],[1.5]])
        #         # zero_gravity = np.zeros((6,1))
        #         # Leg_gravity = np.vstack((zero_gravity,R_RSS_gravity))
        #         Leg_gravity =  0.15*DS_gravity+0.85*LSS_gravity
        #     else:
        #         kl = np.array([[1.2],[1.2],[1.2],[1.5],[1.5],[1.5]])
        #         kr = np.array([[1],[1],[1],[1],[1],[1]])
        #         Leg_gravity =  0.15*DS_gravity+0.85*LSS_gravity
        # else:
        #     kr = np.array([[0.8],[0.8],[0.8],[0.8],[0],[0]])
        #     kl = np.array([[0.8],[0.8],[0.8],[0.8],[0],[0]])
        #     zero_gravity = np.zeros((6,1))
        #     Leg_gravity = np.vstack((zero_gravity, zero_gravity))

        if stance == 0:
            if r_contact == 1:
                kr = np.array([[1.2],[1.2],[1.2],[1.5],[1.5],[1.5]])
            else:
                kr = np.array([[1.2],[1.2],[1.2],[1.2],[1.2],[1.2]])
            kl = np.array([[1],[1],[1],[1],[0],[0]])
            zero_gravity = np.zeros((12,1))
            # Leg_gravity = 0.25*zero_gravity+0.75*RSS_gravity
            Leg_gravity = 0.3*DS_gravity+0.75*RSS_gravity
        
        elif stance == 1:
            kr = np.array([[1],[1],[1],[1],[0],[0]])
            if l_contact == 1:
                kl = np.array([[1.2],[1.2],[1.2],[1.5],[1.5],[1.5]])
            else:
                kl = np.array([[1.2],[1.2],[1.2],[1.2],[1.2],[1.2]])
            zero_gravity = np.zeros((12,1))
            # Leg_gravity =  0.25*zero_gravity+0.75*LSS_gravity
            Leg_gravity =  0.3*DS_gravity+0.75*LSS_gravity

        l_leg_gravity = np.reshape(Leg_gravity[0:6,0],(6,1))
        r_leg_gravity = np.reshape(Leg_gravity[6:,0],(6,1))

        self.l_gravity_publisher.publish(Float64MultiArray(data=l_leg_gravity))
        self.r_gravity_publisher.publish(Float64MultiArray(data=r_leg_gravity))
        
        return l_leg_gravity,r_leg_gravity,kl,kr
    
    def alip_test(self,joint_position,joint_velocity,l_leg_vcmd,r_leg_vcmd,l_leg_gravity_compensate,r_leg_gravity_compensate,kl,kr,px_in_lf):
        print("alip_test")
        jp = copy.deepcopy(joint_position)
        jv = copy.deepcopy(joint_velocity)
        vl_cmd = copy.deepcopy(l_leg_vcmd)
        vr_cmd = copy.deepcopy(r_leg_vcmd)
        l_leg_gravity = copy.deepcopy(l_leg_gravity_compensate)
        r_leg_gravity = copy.deepcopy(r_leg_gravity_compensate)
        l_foot_in_wf = np.array([[0.0],[0.1],[0],[0],[0],[0]]) #平踏於地面時的位置
        px_in_wf = px_in_lf + l_foot_in_wf

        torque = np.zeros((12,1))

        torque[0,0] = kl*(vl_cmd[0,0]-jv[0,0]) + l_leg_gravity[0,0]
        torque[1,0] = kl*(vl_cmd[1,0]-jv[1,0]) + l_leg_gravity[1,0]
        torque[2,0] = kl*(vl_cmd[2,0]-jv[2,0]) + l_leg_gravity[2,0]
        torque[3,0] = kl*(vl_cmd[3,0]-jv[3,0]) + l_leg_gravity[3,0]
        torque[4,0] = 0
        torque[5,0] = 20*(0.2-jp[5,0]) + l_leg_gravity[5,0]

        torque[6,0] = kr*(vr_cmd[0,0]-jv[6,0]) + r_leg_gravity[0,0]
        torque[7,0] = kr*(vr_cmd[1,0]-jv[7,0])+ r_leg_gravity[1,0]
        torque[8,0] = kr*(vr_cmd[2,0]-jv[8,0]) + r_leg_gravity[2,0]
        torque[9,0] = kr*(vr_cmd[3,0]-jv[9,0]) + r_leg_gravity[3,0]
        torque[10,0] = kr*(vr_cmd[4,0]-jv[10,0]) + r_leg_gravity[4,0]
        torque[11,0] = kr*(vr_cmd[5,0]-jv[11,0]) + r_leg_gravity[5,0]

        collect_data = [str(px_in_wf[0,0])]
        csv_file_name = '/home/ldsc/com_x.csv'
        with open(csv_file_name, 'a', newline='') as csvfile:
            # Create a CSV writer object
            csv_writer = csv.writer(csvfile)
            # Write the data
            csv_writer.writerow(collect_data)
        print(f'Data has been written to {csv_file_name}.')

        return torque

    def foot_data(self,px_in_lf,px_in_rf,L,torque_L,com_in_lf):
        l_foot_in_wf = np.array([[0.0],[0.1],[0],[0],[0],[0]]) #平踏於地面時的位置
        r_foot_in_wf = np.array([[0.007],[-0.1],[0],[0],[0],[0]]) #平踏於地面時的位置
        px_in_wf = px_in_lf + l_foot_in_wf
        # px_in_wf = px_in_lf + r_foot_in_wf
        ref_data = np.array([[self.PX_ref[0,0]],[self.PX_ref[1,0]],[self.PX_ref[2,0]],[self.PX_ref[3,0]],[self.PX_ref[4,0]],[self.PX_ref[5,0]]])
        self.ref_publisher.publish(Float64MultiArray(data=ref_data))
        
        # pelvis_data = np.array([[px_in_wf[0,0]],[px_in_wf[1,0]],[px_in_wf[2,0]],[px_in_wf[3,0]],[px_in_wf[4,0]],[px_in_wf[5,0]]])
        pelvis_data = np.array([[px_in_lf[0,0]],[px_in_lf[1,0]],[px_in_lf[2,0]],[px_in_lf[3,0]],[px_in_lf[4,0]],[px_in_lf[5,0]]])
        self.pelvis_publisher.publish(Float64MultiArray(data=pelvis_data))

        if self.state == 9: #ALIP_X實驗
            collect_data = [str(self.PX_ref[0,0]),str(self.PX_ref[1,0]),str(self.PX_ref[2,0]),str(self.PX_ref[3,0]),str(self.PX_ref[4,0]),str(self.PX_ref[5,0]),
            str(px_in_wf[0,0]),str(px_in_wf[1,0]),str(px_in_wf[2,0]),str(px_in_wf[3,0]),str(px_in_wf[4,0]),str(px_in_wf[5,0])]
            csv_file_name = '/home/ldsc/pelvis.csv'
            with open(csv_file_name, 'a', newline='') as csvfile:
                # Create a CSV writer object
                csv_writer = csv.writer(csvfile)
                # Write the data
                csv_writer.writerow(collect_data)
            print(f'Data has been written to {csv_file_name}.')

        if self.state == 7: #ALIP_L平衡實驗
            collect_data = [str(px_in_lf[0,0]),str(px_in_lf[1,0]),str(px_in_lf[2,0]),str(px_in_lf[3,0]),str(px_in_lf[4,0]),str(px_in_lf[5,0])
                            ,str(torque_L[4,0]),str(torque_L[5,0])]
            csv_file_name = '/home/ldsc/impulse_test.csv'
            with open(csv_file_name, 'a', newline='') as csvfile:
                # Create a CSV writer object
                csv_writer = csv.writer(csvfile)
                # Write the data
                csv_writer.writerow(collect_data)
            print(f'Data has been written to {csv_file_name}.')

        if self.state == 4: #ALIP_L質心軌跡追蹤實驗
            collect_data = [str(px_in_lf[2,0]),str(px_in_lf[3,0]),str(px_in_lf[4,0]),str(px_in_lf[5,0]),str(torque_L[4,0]),str(torque_L[5,0])]
            # csv_file_name = '/home/ldsc/alip_tracking_attitude.csv'
            # with open(csv_file_name, 'a', newline='') as csvfile:
            #     # Create a CSV writer object
            #     csv_writer = csv.writer(csvfile)
            #     # Write the data
            #     csv_writer.writerow(collect_data)
            # print(f'Data has been written to {csv_file_name}.')

    def to_matlab(self):
        #only x_y_z
        P_PV_wf = copy.deepcopy(self.P_PV_wf)
        P_COM_wf = copy.deepcopy(self.P_COM_wf)
        P_L_wf = copy.deepcopy(self.P_L_wf)
        P_R_wf = copy.deepcopy(self.P_R_wf)
        
        self.PX_publisher.publish(Float64MultiArray(data=P_PV_wf))
        self.COM_publisher.publish(Float64MultiArray(data=P_COM_wf))
        self.LX_publisher.publish(Float64MultiArray(data=P_L_wf))
        self.RX_publisher.publish(Float64MultiArray(data=P_R_wf))

    def main_controller_callback(self):
        self.P_B_wf, self.O_wfB, self.pub_state, self.l_contact, self.r_contact, joint_position = self.ros.getSubDate()
        
        joint_position,joint_velocity = self.collect_joint_data()
        joint_velocity_cal = self.joint_velocity_cal(joint_position)
        jv_f = self.joint_velocity_filter(joint_velocity_cal)

        configuration = pink.Configuration(self.robot.model, self.robot.data,joint_position)
        self.viz.display(configuration.q)

        #從pink拿相對base_frame的位置及姿態角  ////我覺得是相對pf吧
        self.get_position_pf(configuration)
        px_in_lf,px_in_rf = self.get_posture()
        com_in_lf,com_in_rf,com_in_pink = self.com_position(joint_position)
        #算wf下的位置及姿態
        self.pelvis_in_wf()
        self.data_in_wf(com_in_pink)
        #這邊算相對的矩陣
        self.rotation_matrix(joint_position)
        #這邊算wf下各軸姿態
        self.relative_axis()

        state = self.state_collect()

        l_contact,r_contact = self.contact_collect()

        if self.P_L_wf[2,0] <= 0.01:##\\\\接觸的判斷是z方向在不在0.01以內
            l_contact == 1
        else:
            l_contact == 0
        if self.P_R_wf[2,0] <= 0.01:
            r_contact == 1
        else:
            r_contact == 0

        #========怎麼切支撐狀態要改========!!!!!#
        stance = self.stance_change(state,px_in_lf,px_in_rf,self.stance,self.contact_t)
        
        #========軌跡規劃========#
        self.PX_ref, self.LX_ref, self.RX_ref = trajRef_planning(state, self.DS_time, self.DDT)

        #================#
        l_leg_gravity,r_leg_gravity,kl,kr = self.gravity_compemsate(joint_position,stance,px_in_lf,px_in_rf,l_contact,r_contact,state)
        #========膝上雙環控制========#
        #--------膝上外環控制--------#
        JLL = self.left_leg_jacobian()
        JRR = self.right_leg_jacobian()
        Le_2,Re_2 = endErr_to_endVel(self)
        VL, VR = endVel_to_jv(Le_2,Re_2,jv_f,stance,state,JLL,JRR)
        
        #--------膝上內環控制--------#
        
        #========腳踝ALIP、PD控制========#
        self.checkpub(VL,jv_f)
        
        cf, sf = ('lf','rf') if stance == 1 else \
             ('rf','lf') # if stance == 0, 2
             
        if state == 0:
            torque = balance(joint_position,l_leg_gravity,r_leg_gravity)
            self.ros.publisher['effort'].publish(Float64MultiArray(data=torque))

        elif state == 1:
            torque = innerloopDynamics(jv_f,VL,VR,l_leg_gravity,r_leg_gravity,kl,kr)
            
            torque[sf][4:6] = swingAnkle_PDcontrol(stance, self.O_wfL, self.O_wfR)
            torque[cf][4:6] = alip_control(self.frame, stance, self.stance_past, self.P_COM_wf, self.P_L_wf, self.P_R_wf, self.PX_ref, self.LX_ref,self.RX_ref)
            if stance == 1:
                self.ros.publisher["torque_l"].publish( Float64MultiArray(data = torque['lf'][4:6] ))
            # torque_R =  alip_R(self, stance,px_in_lf,torque_ALIP,com_in_rf,state)
            self.ros.publisher['effort'].publish(Float64MultiArray(data = np.vstack(( torque['lf'], torque['rf'] )) ) )
            
        elif state == 2:
            torque = innerloopDynamics(jv_f,VL,VR,l_leg_gravity,r_leg_gravity,kl,kr)
            
            torque[sf][4:6] = swingAnkle_PDcontrol(stance, self.O_wfL, self.O_wfR)
            torque[cf][4:6] = alip_control(self.frame, stance, self.stance_past, self.P_COM_wf, self.P_L_wf, self.P_R_wf, self.PX_ref, self.LX_ref,self.RX_ref)
            if stance == 1:
                self.ros.publisher["torque_l"].publish( Float64MultiArray(data = torque['lf'][4:6] ))
            # torque_R =  alip_R(self, stance,px_in_lf,torque_ALIP,com_in_rf,state)
            self.ros.publisher['effort'].publish(Float64MultiArray(data = np.vstack(( torque['lf'], torque['rf'] )) ) )

        elif state == 30:
            # self.to_matlab()
            torque_ALIP = walking_by_ALIP(self, jv_f, VL, VR, l_leg_gravity, r_leg_gravity, kl, kr, self.O_wfL, self.O_wfR)
            torque_L =  alip_L(self, stance, torque_ALIP, self.PX_ref, self.LX_ref)
            torque_R =  alip_R(self, stance,px_in_lf,torque_ALIP,com_in_rf,state)
            # print(stance)
            
            if stance == 1:
                self.ros.publisher['effort'].publish(Float64MultiArray(data=torque_L))

            elif stance == 0:
                self.ros.publisher['effort'].publish(Float64MultiArray(data=torque_R))
            # self.ros.publisher['effort'].publish(Float64MultiArray(data=torque_ALIP))
        
        self.state_past = copy.deepcopy(state)
        self.stance_past = copy.deepcopy(stance)
 
    def checkpub(self,VL,jv_f):
        
        self.ros.publisher["vcmd"].publish( Float64MultiArray(data = VL) )
        self.ros.publisher["velocity"].publish( Float64MultiArray(data = jv_f[:6]) )#檢查收到的速度(超髒)

def main(args=None):
    rclpy.init(args=args)

    upper_level_controllers = UpperLevelController()
    rclpy.spin(upper_level_controllers)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    upper_level_controllers.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
