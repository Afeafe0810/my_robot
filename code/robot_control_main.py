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
from utils.config import Config
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
        self.frame = RobotFrame() # 各部位的位置與姿態
        
        self.robot = self.ros.meshrobot
        self.bipedal_floating_model, self.bipedal_floating_data = self.ros.bipedal_floating_model, self.ros.bipedal_floating_data
        self.stance_l_model, self.stance_l_data = self.ros.stance_l_model, self.ros.stance_l_data
        self.stance_r_model, self.stance_r_data = self.ros.stance_r_model, self.ros.stance_r_data
        self.bipedal_l_model, self.bipedal_l_data = self.ros.bipedal_l_model, self.ros.bipedal_l_data
        self.bipedal_r_model, self.bipedal_r_data = self.ros.bipedal_r_model, self.ros.bipedal_r_data
        #==============================================================robot interface==============================================================#
        #joint_velocity_cal
        self.joint_position_past = np.zeros((12,1))

        #joint_velocity_filter (jp = after filter)
        self.jp = np.zeros((12,1))
        self.jv = np.zeros((12,1))
        
        #==============================================================robot constant==============================================================#     
        self.stance = 2
        self.stance_past = 2
        self.DS_time = 0.0

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
            if self.DS_time <= 10 * Config.DDT:
                self.DS_time += Config.TIMER_PERIOD
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

        self.ros.publisher["gravity_l"].publish(Float64MultiArray(data=l_leg_gravity))
        self.ros.publisher["gravity_r"].publish(Float64MultiArray(data=r_leg_gravity))
        
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
        #==========拿取訂閱值==========#
        p_base_in_wf, r_base_to_wf, state, contact_lf, contact_rf, jp, jv = self.ros.updateSubData()
        
        #==========更新可視化的機器人==========#
        config = self.ros.update_VizAndMesh(jp)
        
        #==========更新frame==========#
        # self.frame.updateFrame(config)
        (
            ( self.P_PV_pf , self.O_pfPV  ),
            ( self.P_Lhr_pf, self.O_pfLhr ),
            ( self.P_Lhy_pf, self.O_pfLhy ),
            ( self.P_Lhp_pf, self.O_pfLhp ),
            ( self.P_Lkp_pf, self.O_pfLkp ),
            ( self.P_Lap_pf, self.O_pfLap ),
            ( self.P_Lar_pf, self.O_pfLar ),
            ( self.P_L_pf  , self.O_pfL   ),
            ( self.P_Rhr_pf, self.O_pfRhr ),
            ( self.P_Rhy_pf, self.O_pfRhy ),
            ( self.P_Rhp_pf, self.O_pfRhp ),
            ( self.P_Rkp_pf, self.O_pfRkp ),
            ( self.P_Rap_pf, self.O_pfRap ),
            ( self.P_Rar_pf, self.O_pfRar ),
            ( self.P_R_pf  , self.O_pfR   ),
        ) = self.frame.update_pfFrame(config)
        
        px_in_lf,px_in_rf, self.PX, self.LX, self.RX, self.L_Body_transfer, self.R_Body_transfer = self.frame.get_posture()
        #==========待刪掉==========#
        self.P_B_wf, self.O_wfB, self.pub_state, self.l_contact, self.r_contact, self.jp_sub = p_base_in_wf, r_base_to_wf, state, contact_lf, contact_rf, jp
        l_contact,r_contact = self.l_contact, self.r_contact
        
        

        

        #從pink拿相對base_frame的位置及姿態角  ////我覺得是相對pf吧
        px_in_lf,px_in_rf = self.get_posture()
        com_in_lf,com_in_rf,com_in_pink = self.com_position(jp)
        #算wf下的位置及姿態
        self.pelvis_in_wf()
        self.data_in_wf(com_in_pink)
        #這邊算相對的矩陣
        self.rotation_matrix(jp)
        #這邊算wf下各軸姿態
        self.relative_axis()

        

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
        self.PX_ref, self.LX_ref, self.RX_ref = trajRef_planning(state, self.DS_time, Config.DDT)

        #================#
        l_leg_gravity,r_leg_gravity,kl,kr = self.gravity_compemsate(jp,stance,px_in_lf,px_in_rf,l_contact,r_contact,state)
        #========膝上雙環控制========#
        #--------膝上外環控制--------#
        JLL = self.left_leg_jacobian()
        JRR = self.right_leg_jacobian()
        Le_2,Re_2 = endErr_to_endVel(self)
        VL, VR = endVel_to_jv(Le_2,Re_2,jv,stance,state,JLL,JRR)
        
        #--------膝上內環控制--------#
        
        #========腳踝ALIP、PD控制========#
        self.checkpub(VL,jv)
        
        cf, sf = ('lf','rf') if stance == 1 else \
             ('rf','lf') # if stance == 0, 2
             
        if state == 0:
            torque = balance(jp,l_leg_gravity,r_leg_gravity)
            self.ros.publisher['effort'].publish(Float64MultiArray(data=torque))

        elif state == 1:
            torque = innerloopDynamics(jv,VL,VR,l_leg_gravity,r_leg_gravity,kl,kr)
            
            torque[sf][4:6] = swingAnkle_PDcontrol(stance, self.O_wfL, self.O_wfR)
            torque[cf][4:6] = alip_control(self.frame, stance, self.stance_past, self.P_COM_wf, self.P_L_wf, self.P_R_wf, self.PX_ref, self.LX_ref,self.RX_ref)
            if stance == 1:
                self.ros.publisher["torque_l"].publish( Float64MultiArray(data = torque['lf'][4:6] ))
            # torque_R =  alip_R(self, stance,px_in_lf,torque_ALIP,com_in_rf,state)
            self.ros.publisher['effort'].publish(Float64MultiArray(data = np.vstack(( torque['lf'], torque['rf'] )) ) )
            
        elif state == 2:
            torque = innerloopDynamics(jv,VL,VR,l_leg_gravity,r_leg_gravity,kl,kr)
            
            torque[sf][4:6] = swingAnkle_PDcontrol(stance, self.O_wfL, self.O_wfR)
            torque[cf][4:6] = alip_control(self.frame, stance, self.stance_past, self.P_COM_wf, self.P_L_wf, self.P_R_wf, self.PX_ref, self.LX_ref,self.RX_ref)
            if stance == 1:
                self.ros.publisher["torque_l"].publish( Float64MultiArray(data = torque['lf'][4:6] ))
            # torque_R =  alip_R(self, stance,px_in_lf,torque_ALIP,com_in_rf,state)
            self.ros.publisher['effort'].publish(Float64MultiArray(data = np.vstack(( torque['lf'], torque['rf'] )) ) )

        elif state == 30:
            # self.to_matlab()
            torque_ALIP = walking_by_ALIP(self, jv, VL, VR, l_leg_gravity, r_leg_gravity, kl, kr, self.O_wfL, self.O_wfR)
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
 
    def checkpub(self,VL,jv):
        
        self.ros.publisher["vcmd"].publish( Float64MultiArray(data = VL) )
        self.ros.publisher["velocity"].publish( Float64MultiArray(data = jv[:6]) )#檢查收到的速度(超髒)

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
