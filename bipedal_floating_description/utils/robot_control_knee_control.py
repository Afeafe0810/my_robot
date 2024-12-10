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
class Outterloop:
    @staticmethod
    def left_leg_jacobian(node):
        pelvis = copy.deepcopy(node.P_PV_pf)
        l_hip_roll = copy.deepcopy(node.P_Lhr_pf)
        l_hip_yaw = copy.deepcopy(node.P_Lhy_pf)
        l_hip_pitch = copy.deepcopy(node.P_Lhp_pf)
        l_knee_pitch = copy.deepcopy(node.P_Lkp_pf)
        l_ankle_pitch = copy.deepcopy(node.P_Lap_pf)
        l_ankle_roll = copy.deepcopy(node.P_Lar_pf)
        l_foot = copy.deepcopy(node.P_L_pf)

        JL1 = np.cross(node.AL1,(l_foot-l_hip_roll),axis=0)
        JL2 = np.cross(node.AL2,(l_foot-l_hip_yaw),axis=0)
        JL3 = np.cross(node.AL3,(l_foot-l_hip_pitch),axis=0)
        JL4 = np.cross(node.AL4,(l_foot-l_knee_pitch),axis=0)
        JL5 = np.cross(node.AL5,(l_foot-l_ankle_pitch),axis=0)
        JL6 = np.cross(node.AL6,(l_foot-l_ankle_roll),axis=0)

        JLL_upper = np.hstack((JL1, JL2,JL3,JL4,JL5,JL6))
        JLL_lower = np.hstack((node.AL1,node.AL2,node.AL3,node.AL4,node.AL5,node.AL6))    
        node.JLL = np.vstack((JLL_upper,JLL_lower))  
        # print(node.JLL)

        #排除支撐腳腳踝對末端速度的影響
        node.JL_sp44 = np.reshape(node.JLL[2:,0:4],(4,4))  
        node.JL_sp42 = np.reshape(node.JLL[2:,4:],(4,2))
        #排除擺動.腳踝對末端速度的影響
        JL_sw34 = np.reshape(node.JLL[0:3,0:4],(3,4)) 
        JL_sw14 = np.reshape(node.JLL[5,0:4],(1,4)) 
        node.JL_sw44 = np.vstack((JL_sw34,JL_sw14))

        JL_sw32 = np.reshape(node.JLL[0:3,4:],(3,2)) 
        JL_sw12 = np.reshape(node.JLL[5,4:],(1,2)) 
        node.JL_sw42 = np.vstack((JL_sw32,JL_sw12))

        return node.JLL

    @staticmethod
    def right_leg_jacobian(node):
        pelvis = copy.deepcopy(node.P_PV_pf)
        r_hip_roll = copy.deepcopy(node.P_Rhr_pf)
        r_hip_yaw = copy.deepcopy(node.P_Rhy_pf)
        r_hip_pitch = copy.deepcopy(node.P_Rhp_pf)
        r_knee_pitch = copy.deepcopy(node.P_Rkp_pf)
        r_ankle_pitch = copy.deepcopy(node.P_Rap_pf)
        r_ankle_roll = copy.deepcopy(node.P_Rar_pf)
        r_foot = copy.deepcopy(node.P_R_pf)

        JR1 = np.cross(node.AR1,(r_foot-r_hip_roll),axis=0)
        JR2 = np.cross(node.AR2,(r_foot-r_hip_yaw),axis=0)
        JR3 = np.cross(node.AR3,(r_foot-r_hip_pitch),axis=0)
        JR4 = np.cross(node.AR4,(r_foot-r_knee_pitch),axis=0)
        JR5 = np.cross(node.AR5,(r_foot-r_ankle_pitch),axis=0)
        JR6 = np.cross(node.AR6,(r_foot-r_ankle_roll),axis=0)

        JRR_upper = np.hstack((JR1,JR2,JR3,JR4,JR5,JR6))
        JRR_lower = np.hstack((node.AR1,node.AR2,node.AR3,node.AR4,node.AR5,node.AR6))    
        node.JRR = np.vstack((JRR_upper,JRR_lower))  
        # print(node.JRR)

        #排除支撐腳腳踝對末端速度的影響
        node.JR_sp44 = np.reshape(node.JRR[2:,0:4],(4,4))  
        node.JR_sp42 = np.reshape(node.JRR[2:,4:],(4,2))
        #排除擺動.腳踝對末端速度的影響
        JR_sw34 = np.reshape(node.JRR[0:3,0:4],(3,4)) 
        JR_sw14 = np.reshape(node.JRR[5,0:4],(1,4)) 
        node.JR_sw44 = np.vstack((JR_sw34,JR_sw14))

        JR_sw32 = np.reshape(node.JRR[0:3,4:],(3,2)) 
        JR_sw12 = np.reshape(node.JRR[5,4:],(1,2)) 
        node.JR_sw42 = np.vstack((JR_sw32,JR_sw12))

        return node.JRR

    @staticmethod
    def calculate_err(node,state):
        PX_ref = copy.deepcopy(node.PX_ref) #wf
        LX_ref = copy.deepcopy(node.LX_ref) #wf
        RX_ref = copy.deepcopy(node.RX_ref) #wf
        PX = copy.deepcopy(node.PX) #pf
        LX = copy.deepcopy(node.LX) #pf
        RX = copy.deepcopy(node.RX) #pf
        state = copy.deepcopy(state)
        
        #foot_trajectory(by mynode)
        L_ref = LX_ref - PX_ref 
        R_ref = RX_ref - PX_ref

        L = LX - PX
        R = RX - PX 
        Le = L_ref - L
        Re = R_ref - R
        # --P
        Le_dot = 20*Le
        Re_dot = 20*Re

        # #--PI
        # Le_dot = node.Le_dot_past + 20*Le - 19.99*node.Le_past 
        # node.Le_dot_past = Le_dot
        # node.Le_past = Le

        # Re_dot = node.Re_dot_past + 20*Re - 19.99*node.Re_past 
        # node.Re_dot_past = Re_dot
        # node.Re_past = Re


        Lroll_error_dot = Le_dot[3,0]
        Lpitch_error_dot = Le_dot[4,0]
        Lyaw_error_dot = Le_dot[5,0]
        WL_x = node.L_Body_transfer[0,0]*Lroll_error_dot + node.L_Body_transfer[0,1]*Lpitch_error_dot
        WL_y = node.L_Body_transfer[1,0]*Lroll_error_dot + node.L_Body_transfer[1,1]*Lpitch_error_dot
        WL_z = node.L_Body_transfer[2,0]*Lroll_error_dot + node.L_Body_transfer[2,2]*Lyaw_error_dot

        Le_2 = np.array([[Le_dot[0,0]],[Le_dot[1,0]],[Le_dot[2,0]],[WL_x],[WL_y],[WL_z]])

        Rroll_error_dot = Re_dot[3,0]
        Rpitch_error_dot = Re_dot[4,0]
        Ryaw_error_dot = Re_dot[5,0]
        WR_x = node.R_Body_transfer[0,0]*Rroll_error_dot + node.R_Body_transfer[0,1]*Rpitch_error_dot
        WR_y = node.R_Body_transfer[1,0]*Rroll_error_dot + node.R_Body_transfer[1,1]*Rpitch_error_dot
        WR_z = node.R_Body_transfer[2,0]*Rroll_error_dot + node.R_Body_transfer[2,2]*Ryaw_error_dot

        Re_2 = np.array([[Re_dot[0,0]],[Re_dot[1,0]],[Re_dot[2,0]],[WR_x],[WR_y],[WR_z]])

        return Le_2,Re_2
    
class Innerloop:
    @staticmethod
    def gravity_compemsate(node,joint_position,stance_type,px_in_lf,px_in_rf,l_contact,r_contact,state):

        jp_l = np.reshape(copy.deepcopy(joint_position[0:6,0]),(6,1)) #左腳
        jp_r = np.reshape(copy.deepcopy(joint_position[6:,0]),(6,1))  #右腳
        stance = copy.deepcopy((stance_type))
        
        #DS_gravity
        jp_L_DS = np.flip(-jp_l,axis=0)
        jv_L_DS = np.zeros((6,1))
        c_L_DS = np.zeros((6,1))
        L_DS_gravity = np.reshape(-pin.rnea(node.stance_l_model, node.stance_l_data, jp_L_DS,jv_L_DS,(c_L_DS)),(6,1))  
        L_DS_gravity = np.flip(L_DS_gravity,axis=0)

        jp_R_DS = np.flip(-jp_r,axis=0)
        jv_R_DS = np.zeros((6,1))
        c_R_DS = np.zeros((6,1))
        R_DS_gravity = np.reshape(-pin.rnea(node.stance_r_model, node.stance_r_data, jp_R_DS,jv_R_DS,(c_R_DS)),(6,1))  
        R_DS_gravity = np.flip(R_DS_gravity,axis=0)
        DS_gravity = np.vstack((L_DS_gravity, R_DS_gravity))

        #RSS_gravity
        jp_R_RSS = np.flip(-jp_r,axis=0)
        jp_RSS = np.vstack((jp_R_RSS,jp_l))
        jv_RSS = np.zeros((12,1))
        c_RSS = np.zeros((12,1))
        Leg_RSS_gravity = np.reshape(pin.rnea(node.bipedal_r_model, node.bipedal_r_data, jp_RSS,jv_RSS,(c_RSS)),(12,1))  

        L_RSS_gravity = np.reshape(Leg_RSS_gravity[6:,0],(6,1))
        R_RSS_gravity = np.reshape(-Leg_RSS_gravity[0:6,0],(6,1)) #加負號(相對關係)
        R_RSS_gravity = np.flip(R_RSS_gravity,axis=0)
        RSS_gravity = np.vstack((L_RSS_gravity, R_RSS_gravity))

        #LSS_gravity
        jp_L_LSS = np.flip(-jp_l,axis=0)
        jp_LSS = np.vstack((jp_L_LSS,jp_r))
        jv_LSS = np.zeros((12,1))
        c_LSS = np.zeros((12,1))
        Leg_LSS_gravity = np.reshape(pin.rnea(node.bipedal_l_model, node.bipedal_l_data, jp_LSS,jv_LSS,(c_LSS)),(12,1))  

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

        l_leg_gravity = np.reshape(Leg_gravity[0:6,0],(6,1))
        r_leg_gravity = np.reshape(Leg_gravity[6:,0],(6,1))

        node.l_gravity_publisher.publish(Float64MultiArray(data=l_leg_gravity))
        node.r_gravity_publisher.publish(Float64MultiArray(data=r_leg_gravity))
        
        return l_leg_gravity,r_leg_gravity,kl,kr
    
    @staticmethod
    def velocity_cmd(node,Le_2,Re_2,jv_f,stance_type,state):

        L2 = copy.deepcopy(Le_2)
        R2 = copy.deepcopy(Re_2)
        v =  copy.deepcopy(jv_f) #joint_velocity
        state = copy.deepcopy(state)
       
        #獲取支撐狀態
        stance = copy.deepcopy(stance_type)
        # print(stance)
        if state == 1:
            if stance == 0:   
                print('enter stance0')
                #(右支撐腳腳踝動態排除)
                R2_41 = np.reshape(R2[2:,0],(4,1)) #R2 z to wz
                VR56 =  np.reshape(v[10:,0],(2,1)) #右腳腳踝速度
                #計算右膝關節以上速度
                R2_41_cal = R2_41 - node.JR_sp42@VR56
                #彙整右腳速度
                rw_41_d = np.dot(np.linalg.pinv(node.JR_sp44),R2_41_cal)
                rw_21_d = np.zeros((2,1))
                Rw_d = np.vstack((rw_41_d,rw_21_d))

                #(左擺動腳腳踝動態排除)
                #拿左腳 誤差及腳踝速度
                L2_41 = np.array([[L2[0,0]],[L2[1,0]],[L2[2,0]],[L2[5,0]]]) #x y z yaw
                VL56 =  np.reshape(v[4:6,0],(2,1)) #左腳腳踝速度
                #計算左膝關節以上速度
                L2_41_cal = L2_41 - node.JL_sw42@VL56
                #彙整左腳速度
                lw_41_d = np.dot(np.linalg.pinv(node.JL_sw44),L2_41_cal)
                lw_21_d = np.zeros((2,1))
                Lw_d = np.vstack((lw_41_d,lw_21_d))

                # Lw_d = np.dot(np.linalg.pinv(node.JLL),L2) 
                
            elif stance == 1 or stance == 2 :
                print('enter stance1')   
                #(左支撐腳腳踝動態排除)
                #拿左腳 誤差及腳踝速度
                L2_41 = np.reshape(L2[2:,0],(4,1)) #L2 z to wz
                VL56 =  np.reshape(v[4:6,0],(2,1)) #左腳腳踝速度
                #計算左膝關節以上速度
                L2_41_cal = L2_41 - node.JL_sp42@VL56
                #彙整左腳速度
                lw_41_d = np.dot(np.linalg.pinv(node.JL_sp44),L2_41_cal)
                lw_21_d = np.zeros((2,1))
                Lw_d = np.vstack((lw_41_d,lw_21_d))

                #(右擺動腳腳踝動態排除)
                #拿右腳 誤差及腳踝速度
                R2_41 = np.array([[R2[0,0]],[R2[1,0]],[R2[2,0]],[R2[5,0]]]) #x y z yaw
                VR56 =  np.reshape(v[10:,0],(2,1)) #右腳腳踝速度
                #計算右膝關節以上速度
                R2_41_cal = R2_41 - node.JR_sw42@VR56
                #彙整右腳速度
                rw_41_d = np.dot(np.linalg.pinv(node.JR_sw44),R2_41_cal)
                rw_21_d = np.zeros((2,1))
                Rw_d = np.vstack((rw_41_d,rw_21_d))
                # Rw_d = np.dot(np.linalg.pinv(node.JRR),R2) 
        else:
            Lw_d = np.dot(np.linalg.pinv(node.JLL),L2) 
            Rw_d = np.dot(np.linalg.pinv(node.JRR),R2) 

        return Lw_d,Rw_d
    