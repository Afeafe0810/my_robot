#================ import library ========================#
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray 

import pinocchio as pin

import pink

import numpy as np
np.set_printoptions(precision=2)

from sys import argv
from os.path import dirname, join, abspath
import copy
from math import cos, sin
from scipy.spatial.transform import Rotation as R

import csv

#========================================================#

class ULC_frame:
    def __init__(self):
        self.P_B_wf = self.O_wfB = None
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
        
        #--velocity
        self.CX_past_L = 0.0
        self.CX_dot_L = 0.0
        self.CY_past_L = 0.0
        self.CY_dot_L = 0.0
        self.CX_past_R = 0.0
        self.CX_dot_R = 0.0
        self.CY_past_R = 0.0
        self.CY_dot_R = 0.0
        #--velocity filter
        self.Vx_L = 0.0
        self.Vx_past_L = 0.0
        self.CX_dot_past_L = 0.0
        self.Vy_L = 0.0# Set initial robot configuration
        self.Vy_past_L = 0.0
        self.CY_dot_past_L = 0.0
        self.Vx_R = 0.0
        self.Vx_past_R = 0.0
        self.CX_dot_past_R = 0.0
        self.Vy_R = 0.0
        self.Vy_past_R = 0.0
        self.CY_dot_past_R = 0.0
        #--measurement
        self.mea_x_L = np.zeros((2,1))
        self.mea_x_past_L = np.zeros((2,1))
        self.mea_y_L = np.zeros((2,1))
        self.mea_y_past_L = np.zeros((2,1))
        self.mea_x_R = np.zeros((2,1))
        self.mea_x_past_R = np.zeros((2,1))
        self.mea_y_R = np.zeros((2,1))
        self.mea_y_past_R = np.zeros((2,1))
        
        pass
    
    @staticmethod
    def rotationToAngle(mat):
        return np.vstack(( R.from_matrix(mat).as_euler('zyx', degrees=False)[::-1] ))
    
    def get_position_pf(self, config:pink.Configuration):
        
        self.P_PV_pf,  self.O_pfPV  = self.__linkToPf_by_config(config, "pelvis_link")
        self.P_Lhr_pf, self.O_pfLhr = self.__linkToPf_by_config(config, "l_hip_yaw_1")
        self.P_Lhy_pf, self.O_pfLhy = self.__linkToPf_by_config(config, "l_hip_pitch_1")
        self.P_Lhp_pf, self.O_pfLhp = self.__linkToPf_by_config(config, "l_thigh_1")
        self.P_Lkp_pf, self.O_pfLkp = self.__linkToPf_by_config(config, "l_shank_1")
        self.P_Lap_pf, self.O_pfLap = self.__linkToPf_by_config(config, "l_ankle_1")
        self.P_Lar_pf, self.O_pfLar = self.__linkToPf_by_config(config, "l_foot_1")
        self.P_L_pf,   self.O_pfL   = self.__linkToPf_by_config(config, "l_foot")
        
        self.P_Rhr_pf, self.O_pfRhr = self.__linkToPf_by_config(config, "r_hip_yaw_1")
        self.P_Rhy_pf, self.O_pfRhy = self.__linkToPf_by_config(config, "r_hip_pitch_1")
        self.P_Rhp_pf, self.O_pfRhp = self.__linkToPf_by_config(config, "r_thigh_1")
        self.P_Rkp_pf, self.O_pfRkp = self.__linkToPf_by_config(config, "r_shank_1")
        self.P_Rap_pf, self.O_pfRap = self.__linkToPf_by_config(config, "r_ankle_1")
        self.P_Rar_pf, self.O_pfRar = self.__linkToPf_by_config(config, "r_foot_1")
        self.P_R_pf,   self.O_pfR   = self.__linkToPf_by_config(config, "r_foot")
        
    @staticmethod
    def __linkToPf_by_config(config:pink.Configuration, link):
        """ """
        frame = config.get_transform_frame_to_world(link)
        p_frame_in_pf = np.reshape(frame.translation,(3,1))
        r_frame2pf = np.reshape(frame.rotation,(3,3))
        return p_frame_in_pf, r_frame2pf

    def get_posture(self):
        
        ##////把旋轉矩陣換成歐拉角zyx
        pR = self.rotationToAngle(self.O_pfPV)
        lR = self.rotationToAngle(self.O_pfL)
        rR = self.rotationToAngle(self.O_pfR)

        self.PX = np.vstack(( self.P_PV_pf, pR ))
        self.LX = np.vstack(( self.P_L_pf, lR ))
        self.RX = np.vstack(( self.P_R_pf, rR ))
        

        ##////這是啥
        self.L_Body_transfer = np.array([
            [cos(lR[1])*cos(lR[2]), -sin(lR[2]), 0],
            [cos(lR[1])*sin(lR[2]),  cos(lR[2]), 0],
            [-sin(lR[1]),            0,          1]
        ])  
        
        self.R_Body_transfer = np.array([
            [cos(rR[1])*cos(rR[2]), -sin(rR[2]), 0],
            [cos(rR[1])*sin(rR[2]),  cos(rR[2]), 0],
            [-sin(rR[1]),            0,          1]
        ])  
        
        px_in_lf = self.PX - self.LX #骨盆中心相對於左腳
        px_in_rf = self.PX - self.RX #骨盆中心相對於右腳

        return px_in_lf,px_in_rf

    @staticmethod
    def com_position(ulc: Node,joint_position):
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
        pin.centerOfMass(ulc.bipedal_r_model,ulc.bipedal_r_data,R_joint_angle)
        com_r_in_pink = np.reshape(ulc.bipedal_r_data.com[0],(3,1))
        r_foot_in_wf = np.array([[0.0],[-0.1],[0.0]]) ##////其實是pink，原點在兩隻腳正中間
        com_in_rf = com_r_in_pink - r_foot_in_wf

        #左腳為支撐腳
        L_jp_l = np.flip(-jp_l,axis=0)
        L_jp_r = jp_r
        L_joint_angle = np.vstack((L_jp_l,L_jp_r))
        pin.centerOfMass(ulc.bipedal_l_model,ulc.bipedal_l_data,L_joint_angle)
        com_l_in_pink = np.reshape(ulc.bipedal_l_data.com[0],(3,1))
        l_foot_in_wf = np.array([[0.0],[0.1],[0]])
        com_in_lf = com_l_in_pink - l_foot_in_wf

        #floating com ////是從骨盆中建，所以應該可以得知骨盆和質心的位置吧？？
        joint_angle = np.vstack((jp_l,jp_r))
        pin.centerOfMass(ulc.bipedal_floating_model,ulc.bipedal_floating_data,joint_angle)
        com_floating_in_pink = np.reshape(ulc.bipedal_floating_data.com[0],(3,1))

        return com_in_lf,com_in_rf,com_floating_in_pink
    
    def pelvis_in_wf(self):
        '''
        用訂閱到的base對WF的位態,求骨盆對WF的位態
        '''
        self.P_PV_wf = self.O_wfB @ np.vstack(( 0, 0, 0.598 )) + self.P_B_wf
        self.O_wfPV = copy.deepcopy(self.O_wfB)

        # node.PX_publisher.publish(Float64MultiArray(data=node.P_PV_wf))
        return 
    
    def data_in_wf(self, com_in_pink:np.ndarray):
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

    @staticmethod
    def joint_velocity_filter(node,joint_velocity):
        '''
        把差分得到的速度做5個點差分，但pp是什麼意思？？是怎麼濾的
        '''
        jv_sub = copy.deepcopy(joint_velocity)

        # node.jv = 1.1580*node.jv_p - 0.4112*node.jv_pp + 0.1453*node.jv_sub_p + 0.1078*node.jv_sub_pp #10Hz
        # node.jv = 0.5186*node.jv_p - 0.1691*node.jv_pp + 0.4215*node.jv_sub_p + 0.229*node.jv_sub_pp #20Hz
        node.jv = 0.0063*node.jv_p - 0.0001383*node.jv_pp + 1.014*node.jv_sub_p -0.008067*node.jv_sub_pp #100Hz

        node.jv_pp = copy.deepcopy(node.jv_p)
        node.jv_p = copy.deepcopy(node.jv)
        node.jv_sub_pp = copy.deepcopy(node.jv_sub_p)
        node.jv_sub_p = copy.deepcopy(jv_sub)

        return node.jv
    
    @staticmethod
    def joint_velocity_cal(node,joint_position):
        '''
        用差分計算速度，並且加上飽和條件[-0.75, 0.75]、更新joint_position_past(感覺沒意義)
        '''
        joint_position_now = copy.deepcopy(joint_position)
        joint_velocity_cal = (joint_position_now - node.joint_position_past)/node.timer_period
        node.joint_position_past = joint_position_now     
        
        joint_velocity_cal = np.reshape(joint_velocity_cal,(12,1))

        for i in range(len(joint_velocity_cal)):
            if joint_velocity_cal[i,0]>= 0.75:
                joint_velocity_cal[i,0] = 0.75
            elif joint_velocity_cal[i,0]<= -0.75:
                joint_velocity_cal[i,0] = -0.75

        return joint_velocity_cal

    @staticmethod
    def xyz_rotation(axis,theta):
        R = np.array((3,3))
        if axis == 'x':
            R = np.array([[1,0,0],[0,cos(theta),-sin(theta)],[0,sin(theta),cos(theta)]])
        elif axis == 'y':
            R = np.array([[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]])
        elif axis == 'z':
            R = np.array([[cos(theta),-sin(theta),0],[sin(theta),cos(theta),0],[0,0,1]])
        return R 