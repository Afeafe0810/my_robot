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
import math
from scipy.spatial.transform import Rotation as R

import csv

#========================================================#

class ULC_sensor:
    
    @staticmethod
    def get_position_pf(node:Node,configuration:pink.Configuration):
        '''
        把各個座標系對pf的原點、旋轉矩陣賦值到node內
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
        # print("l_foot:",node.l_foot.translation)
        Rhr_pf = configuration.get_transform_frame_to_world("r_hip_yaw_1")
        Rhy_pf = configuration.get_transform_frame_to_world("r_hip_pitch_1")
        Rhp_pf = configuration.get_transform_frame_to_world("r_thigh_1")
        Rkp_pf = configuration.get_transform_frame_to_world("r_shank_1")
        Rap_pf = configuration.get_transform_frame_to_world("r_ankle_1")
        Rar_pf = configuration.get_transform_frame_to_world("r_foot_1")
        R_pf = configuration.get_transform_frame_to_world("r_foot")
        # print("r_foot:",node.r_foot.translation)        

        #frame origin position in pf
        node.P_PV_pf = np.reshape(PV_pf.translation,(3,1))

        node.P_Lhr_pf = np.reshape(Lhr_pf.translation,(3,1))
        node.P_Lhy_pf = np.reshape(Lhy_pf.translation,(3,1))
        node.P_Lhp_pf = np.reshape(Lhp_pf.translation,(3,1))
        node.P_Lkp_pf = np.reshape(Lkp_pf.translation,(3,1))
        node.P_Lap_pf = np.reshape(Lap_pf.translation,(3,1))
        node.P_Lar_pf = np.reshape(Lar_pf.translation,(3,1))
        node.P_L_pf = np.reshape(L_pf.translation,(3,1))

        node.P_Rhr_pf = np.reshape(Rhr_pf.translation,(3,1))
        node.P_Rhy_pf = np.reshape(Rhy_pf.translation,(3,1))
        node.P_Rhp_pf = np.reshape(Rhp_pf.translation,(3,1))
        node.P_Rkp_pf = np.reshape(Rkp_pf.translation,(3,1))
        node.P_Rap_pf = np.reshape(Rap_pf.translation,(3,1))
        node.P_Rar_pf = np.reshape(Rar_pf.translation,(3,1))
        node.P_R_pf = np.reshape(R_pf.translation,(3,1))

        #frame orientation in pf
        node.O_pfPV = np.reshape(PV_pf.rotation,(3,3))
        node.O_pfLhr = np.reshape(Lhr_pf.rotation,(3,3))
        node.O_pfLhy = np.reshape(Lhy_pf.rotation,(3,3))
        node.O_pfLhp = np.reshape(Lhp_pf.rotation,(3,3))
        node.O_pfLkp = np.reshape(Lkp_pf.rotation,(3,3))
        node.O_pfLap = np.reshape(Lap_pf.rotation,(3,3))
        node.O_pfLar = np.reshape(Lar_pf.rotation,(3,3))
        node.O_pfL = np.reshape(L_pf.rotation,(3,3))

        node.O_pfRhr = np.reshape(Rhr_pf.rotation,(3,3))
        node.O_pfRhy = np.reshape(Rhy_pf.rotation,(3,3))
        node.O_pfRhp = np.reshape(Rhp_pf.rotation,(3,3))
        node.O_pfRkp = np.reshape(Rkp_pf.rotation,(3,3))
        node.O_pfRap = np.reshape(Rap_pf.rotation,(3,3))
        node.O_pfRar = np.reshape(Rar_pf.rotation,(3,3))
        node.O_pfR = np.reshape(R_pf.rotation,(3,3))

    @staticmethod
    def get_posture(node):
        '''
        回傳(骨盆相對於左腳，骨盆相對於右腳)，但body transfer不知道是什麼
        '''
        cos = math.cos
        sin = math.sin

        pelvis_p = copy.deepcopy(node.P_PV_pf)
        l_foot_p = copy.deepcopy(node.P_L_pf)
        r_foot_p = copy.deepcopy(node.P_R_pf)

        # pelvis_p = copy.deepcopy(node.P_PV_wf)
        # l_foot_p = copy.deepcopy(node.P_L_wf)
        # r_foot_p = copy.deepcopy(node.P_R_wf)

        pelvis_o = copy.deepcopy(node.O_pfPV)
        l_foot_o = copy.deepcopy(node.O_pfL)
        r_foot_o = copy.deepcopy(node.O_pfR)

        # pelvis_o = copy.deepcopy(node.O_wfPV)
        # l_foot_o = copy.deepcopy(node.O_wfL)
        # r_foot_o = copy.deepcopy(node.O_wfR)

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

        node.PX = np.array([[pelvis_p[0,0]],[pelvis_p[1,0]],[pelvis_p[2,0]],[P_Roll],[P_Pitch],[P_Yaw]])
        node.LX = np.array([[l_foot_p[0,0]],[l_foot_p[1,0]],[l_foot_p[2,0]],[L_Roll],[L_Pitch],[L_Yaw]])
        node.RX = np.array([[r_foot_p[0,0]],[r_foot_p[1,0]],[r_foot_p[2,0]],[R_Roll],[R_Pitch],[R_Yaw]])

        ##////這是啥
        node.L_Body_transfer = np.array([[cos(L_Pitch)*cos(L_Yaw), -sin(L_Yaw),0],
                                [cos(L_Pitch)*sin(L_Yaw), cos(L_Yaw), 0],
                                [-sin(L_Pitch), 0, 1]])  
        
        node.R_Body_transfer = np.array([[cos(R_Pitch)*cos(R_Yaw), -sin(R_Yaw),0],
                                [cos(R_Pitch)*sin(R_Yaw), cos(R_Yaw), 0],
                                [-sin(R_Pitch), 0, 1]])  
        
        # print("PX",node.PX)
        # print("LX",node.LX)
        # print("RX",node.RX)

        px_in_lf = node.PX - node.LX #骨盆中心相對於左腳
        px_in_rf = node.PX - node.RX #骨盆中心相對於右腳

        return px_in_lf,px_in_rf

    @staticmethod
    def com_position(node:Node,joint_position):
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
        pin.centerOfMass(node.bipedal_r_model,node.bipedal_r_data,R_joint_angle)
        com_r_in_pink = np.reshape(node.bipedal_r_data.com[0],(3,1))
        r_foot_in_wf = np.array([[0.0],[-0.1],[0.0]]) ##////其實是pink，原點在兩隻腳正中間
        com_in_rf = com_r_in_pink - r_foot_in_wf

        #左腳為支撐腳
        L_jp_l = np.flip(-jp_l,axis=0)
        L_jp_r = jp_r
        L_joint_angle = np.vstack((L_jp_l,L_jp_r))
        pin.centerOfMass(node.bipedal_l_model,node.bipedal_l_data,L_joint_angle)
        com_l_in_pink = np.reshape(node.bipedal_l_data.com[0],(3,1))
        l_foot_in_wf = np.array([[0.0],[0.1],[0]])
        com_in_lf = com_l_in_pink - l_foot_in_wf

        #floating com ////是從骨盆中建，所以應該可以得知骨盆和質心的位置吧？？
        joint_angle = np.vstack((jp_l,jp_r))
        pin.centerOfMass(node.bipedal_floating_model,node.bipedal_floating_data,joint_angle)
        com_floating_in_pink = np.reshape(node.bipedal_floating_data.com[0],(3,1))
        # print(com_floating_in_pink)

        # print('clf:',com_in_lf)
        # print('crf:',com_in_rf)

        return com_in_lf,com_in_rf,com_floating_in_pink
    
    @staticmethod
    def pelvis_in_wf(node:Node):
        '''
        用訂閱到的base對WF的位態,求骨盆對WF的位態
        '''
        P_B_wf = copy.deepcopy(node.P_B_wf)##////base對WF的位置
        O_wfB = copy.deepcopy(node.O_wfB)##////base對WF的旋轉矩陣

        node.P_PV_wf = O_wfB@np.array([[0.0],[0.0],[0.598]]) + P_B_wf
        node.O_wfPV = copy.deepcopy(node.O_wfB)

        # node.PX_publisher.publish(Float64MultiArray(data=node.P_PV_wf))

        return 
    
    @staticmethod
    def data_in_wf(node:Node, com_in_pink):
        '''
        就.....一堆轉換
        '''
        #pf_p
        P_PV_pf = copy.deepcopy(node.P_PV_pf)
        P_COM_pf = copy.deepcopy(com_in_pink)

        P_Lhr_pf = copy.deepcopy(node.P_Lhr_pf)
        P_Lhy_pf = copy.deepcopy(node.P_Lhy_pf)
        P_Lhp_pf = copy.deepcopy(node.P_Lhp_pf)
        P_Lkp_pf = copy.deepcopy(node.P_Lkp_pf)
        P_Lap_pf = copy.deepcopy(node.P_Lap_pf)
        P_Lar_pf = copy.deepcopy(node.P_Lar_pf)

        P_L_pf= copy.deepcopy(node.P_L_pf) 

        P_Rhr_pf = copy.deepcopy(node.P_Rhr_pf)
        P_Rhy_pf = copy.deepcopy(node.P_Rhy_pf)
        P_Rhp_pf = copy.deepcopy(node.P_Rhp_pf)
        P_Rkp_pf = copy.deepcopy(node.P_Rkp_pf)
        P_Rap_pf = copy.deepcopy(node.P_Rap_pf)
        P_Rar_pf = copy.deepcopy(node.P_Rar_pf)

        P_R_pf = copy.deepcopy(node.P_R_pf) 

        #pf_o
        O_pfL = copy.deepcopy(node.O_pfL)
        O_pfR = copy.deepcopy(node.O_pfR)
        
        #PV_o ////直走所以是單位矩陣，沒有旋轉
        O_PVpf = np.identity(3)
        #wf_p
        P_PV_wf = copy.deepcopy(node.P_PV_wf) #ros-3d
        O_wfPV = copy.deepcopy(node.O_wfPV) #ros-3d
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
        node.P_PV_wf = copy.deepcopy(P_PV_wf)
        node.P_COM_wf = copy.deepcopy(P_COM_wf)

        node.P_Lhr_wf = copy.deepcopy(P_Lhr_wf)
        node.P_Lhy_wf = copy.deepcopy(P_Lhy_wf)
        node.P_Lhp_wf = copy.deepcopy(P_Lhp_wf)
        node.P_Lkp_wf = copy.deepcopy(P_Lkp_wf)
        node.P_Lap_wf = copy.deepcopy(P_Lap_wf)
        node.P_Lar_wf = copy.deepcopy(P_Lar_wf)

        node.P_L_wf = copy.deepcopy(P_L_wf)

        node.P_Rhr_wf = copy.deepcopy(P_Rhr_wf)
        node.P_Rhy_wf = copy.deepcopy(P_Rhy_wf)
        node.P_Rhp_wf = copy.deepcopy(P_Rhp_wf)
        node.P_Rkp_wf = copy.deepcopy(P_Rkp_wf)
        node.P_Rap_wf = copy.deepcopy(P_Rap_wf)
        node.P_Rar_wf = copy.deepcopy(P_Rar_wf)

        node.P_R_wf = copy.deepcopy(P_R_wf)
        #orientation in wf
        node.O_wfPV = copy.deepcopy(O_wfPV)
        node.O_wfL = copy.deepcopy(O_wfL)
        node.O_wfR = copy.deepcopy(O_wfR)

        # node.PX_publisher.publish(Float64MultiArray(data=P_PV_wf))
        # node.COM_publisher.publish(Float64MultiArray(data=P_COM_wf))
        # node.LX_publisher.publish(Float64MultiArray(data=P_L_wf))
        # node.RX_publisher.publish(Float64MultiArray(data=P_R_wf))

        return 
    
    @staticmethod
    def rotation_matrix(node:Node,joint_position):
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
        node.L_R01 = node.xyz_rotation('x',Theta1) #L_Hip_roll
        node.L_R12 = node.xyz_rotation('z',Theta2)
        node.L_R23 = node.xyz_rotation('y',Theta3)
        node.L_R34 = node.xyz_rotation('y',Theta4)
        node.L_R45 = node.xyz_rotation('y',Theta5)
        node.L_R56 = node.xyz_rotation('x',Theta6) #L_Ankle_roll

        node.R_R01 = node.xyz_rotation('x',Theta7) #R_Hip_roll
        node.R_R12 = node.xyz_rotation('z',Theta8)
        node.R_R23 = node.xyz_rotation('y',Theta9)
        node.R_R34 = node.xyz_rotation('y',Theta10)
        node.R_R45 = node.xyz_rotation('y',Theta11)
        node.R_R56 = node.xyz_rotation('x',Theta12) #R_Ankle_roll

    @staticmethod
    def relative_axis(node):
        '''
        不知道是幹麻的
        '''
        #骨盆姿態(要確認！)
        node.RP = np.array([[1,0,0],[0,1,0],[0,0,1]])
        # node.RP = copy.deepcopy(node.O_wfPV)

        node.AL1 = node.RP@(np.array([[1],[0],[0]])) #L_Hip_roll
        node.AL2 = node.RP@node.L_R01@(np.array([[0],[0],[1]])) 
        node.AL3 = node.RP@node.L_R01@node.L_R12@(np.array([[0],[1],[0]])) 
        node.AL4 = node.RP@node.L_R01@node.L_R12@node.L_R23@(np.array([[0],[1],[0]]))
        node.AL5 = node.RP@node.L_R01@node.L_R12@node.L_R23@node.L_R34@(np.array([[0],[1],[0]])) 
        node.AL6 = node.RP@node.L_R01@node.L_R12@node.L_R23@node.L_R34@node.L_R45@(np.array([[1],[0],[0]])) #L_Ankle_Roll

        node.AR1 = node.RP@(np.array([[1],[0],[0]])) #R_Hip_roll
        node.AR2 = node.RP@node.R_R01@(np.array([[0],[0],[1]])) 
        node.AR3 = node.RP@node.R_R01@node.R_R12@(np.array([[0],[1],[0]])) 
        node.AR4 = node.RP@node.R_R01@node.R_R12@node.R_R23@(np.array([[0],[1],[0]]))
        node.AR5 = node.RP@node.R_R01@node.R_R12@node.R_R23@node.R_R34@(np.array([[0],[1],[0]])) 
        node.AR6 = node.RP@node.R_R01@node.R_R12@node.R_R23@node.R_R34@node.R_R45@(np.array([[1],[0],[0]])) #R_Ankle_Roll
