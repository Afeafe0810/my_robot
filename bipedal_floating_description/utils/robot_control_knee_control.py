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
        node.lpublisher['l_gravity'].publish(Float64MultiArray(data=l_leg_gravity))
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
    
    @staticmethod
    def balance(node,joint_position,l_leg_gravity_compensate,r_leg_gravity_compensate):
        #balance the robot to initial state by p_control
        jp = copy.deepcopy(joint_position)
        # p = np.array([[0.0],[0.0],[-0.37],[0.74],[-0.37],[0.0],[0.0],[0.0],[-0.37],[0.74],[-0.37],[0.0]])
        p = np.zeros((12,1))
        l_leg_gravity = copy.deepcopy(l_leg_gravity_compensate)
        r_leg_gravity = copy.deepcopy(r_leg_gravity_compensate)

        torque = np.zeros((12,1))
        torque[0,0] = 2*(p[0,0]-jp[0,0]) + l_leg_gravity[0,0]
        torque[1,0] = 2*(p[1,0]-jp[1,0]) + l_leg_gravity[1,0]
        torque[2,0] = 4*(p[2,0]-jp[2,0]) + l_leg_gravity[2,0]
        torque[3,0] = 6*(p[3,0]-jp[3,0]) + l_leg_gravity[3,0]
        torque[4,0] = 6*(p[4,0]-jp[4,0]) + l_leg_gravity[4,0]
        torque[5,0] = 4*(p[5,0]-jp[5,0]) + l_leg_gravity[5,0]

        torque[6,0] = 2*(p[6,0]-jp[6,0]) + r_leg_gravity[0,0]
        torque[7,0] = 2*(p[7,0]-jp[7,0]) + r_leg_gravity[1,0]
        torque[8,0] = 4*(p[8,0]-jp[8,0]) + r_leg_gravity[2,0]
        torque[9,0] = 6*(p[9,0]-jp[9,0]) + r_leg_gravity[3,0]
        torque[10,0] = 6*(p[10,0]-jp[10,0]) + r_leg_gravity[4,0]
        torque[11,0] = 4*(p[11,0]-jp[11,0]) + r_leg_gravity[5,0]

        node.effort_publisher.publish(Float64MultiArray(data=torque))

    @staticmethod
    def swing_leg(node,joint_velocity,l_leg_vcmd,r_leg_vcmd,l_leg_gravity_compensate,r_leg_gravity_compensate,kl,kr):
        print("swing_mode")
        node.tt += 0.0157
        jv = copy.deepcopy(joint_velocity)
        vl_cmd = copy.deepcopy(l_leg_vcmd)
        vr_cmd = copy.deepcopy(r_leg_vcmd)
        l_leg_gravity = copy.deepcopy(l_leg_gravity_compensate)
        r_leg_gravity = copy.deepcopy(r_leg_gravity_compensate)

        # print("com_lf:",com_in_lf)
        # print("com_rf:",com_in_rf)
        # #L_leg_velocity
        # vl = np.reshape(copy.deepcopy(joint_velocity[:6,0]),(6,1))

        torque = np.zeros((12,1))

        torque[0,0] = kl[0,0]*(vl_cmd[0,0]-jv[0,0]) + l_leg_gravity[0,0]
        torque[1,0] = kl[1,0]*(vl_cmd[1,0]-jv[1,0]) + l_leg_gravity[1,0]
        torque[2,0] = kl[2,0]*(vl_cmd[2,0]-jv[2,0]) + l_leg_gravity[2,0]
        torque[3,0] = kl[3,0]*(vl_cmd[3,0]-jv[3,0]) + l_leg_gravity[3,0]
        torque[4,0] = kl[4,0]*(vl_cmd[4,0]-jv[4,0]) + l_leg_gravity[4,0]
        torque[5,0] = kl[5,0]*(vl_cmd[5,0]-jv[5,0]) + l_leg_gravity[5,0]

        torque[6,0] = kr[0,0]*(vr_cmd[0,0]-jv[6,0]) + r_leg_gravity[0,0]
        torque[7,0] = kr[1,0]*(vr_cmd[1,0]-jv[7,0])+ r_leg_gravity[1,0]
        torque[8,0] = kr[2,0]*(vr_cmd[2,0]-jv[8,0]) + r_leg_gravity[2,0]
        torque[9,0] = kr[3,0]*(vr_cmd[3,0]-jv[9,0]) + r_leg_gravity[3,0]
        torque[10,0] = kr[4,0]*(vr_cmd[4,0]-jv[10,0]) + r_leg_gravity[4,0]
        torque[11,0] = kr[5,0]*(vr_cmd[5,0]-jv[11,0]) + r_leg_gravity[5,0]

        # node.effort_publisher.publish(Float64MultiArray(data=torque))
        
        vcmd_data = np.array([[vl_cmd[0,0]],[vl_cmd[1,0]],[vl_cmd[2,0]],[vl_cmd[3,0]],[vl_cmd[4,0]],[vl_cmd[5,0]]])
        node.vcmd_publisher.publish(Float64MultiArray(data=vcmd_data))
        jv_collect = np.array([[jv[0,0]],[jv[1,0]],[jv[2,0]],[jv[3,0]],[jv[4,0]],[jv[5,0]]])
        node.velocity_publisher.publish(Float64MultiArray(data=jv_collect))#檢查收到的速度(超髒)

        return torque

    @staticmethod
    def walking_by_ALIP(node,joint_velocity,l_leg_vcmd,r_leg_vcmd,l_leg_gravity_compensate,r_leg_gravity_compensate,kl,kr):
        # print("ALIP_mode")
        node.tt += 0.0157
        jv = copy.deepcopy(joint_velocity)
        vl_cmd = copy.deepcopy(l_leg_vcmd)
        vr_cmd = copy.deepcopy(r_leg_vcmd)
        l_leg_gravity = copy.deepcopy(l_leg_gravity_compensate)
        r_leg_gravity = copy.deepcopy(r_leg_gravity_compensate)

        L_matrix = R.from_matrix(node.O_wfL)
        L_euler = L_matrix.as_euler('zyx', degrees=False)
        L_yaw = L_euler[0]
        L_pitch = L_euler[1]
        L_roll = L_euler[2]

        R_matrix = R.from_matrix(node.O_wfR)
        R_euler = R_matrix.as_euler('zyx', degrees=False)
        R_yaw = R_euler[0]
        R_pitch = R_euler[1]
        R_roll = R_euler[2]

        # print(L_pitch)

        torque = np.zeros((12,1))

        torque[0,0] = kl[0,0]*(vl_cmd[0,0]-jv[0,0]) + l_leg_gravity[0,0]
        torque[1,0] = kl[1,0]*(vl_cmd[1,0]-jv[1,0]) + l_leg_gravity[1,0]
        torque[2,0] = kl[2,0]*(vl_cmd[2,0]-jv[2,0]) + l_leg_gravity[2,0]
        torque[3,0] = kl[3,0]*(vl_cmd[3,0]-jv[3,0]) + l_leg_gravity[3,0]

        #運動學
        # torque[4,0] = kl[4,0]*(vl_cmd[4,0]-jv[4,0]) + l_leg_gravity[4,0]
        # torque[5,0] = kl[5,0]*(vl_cmd[5,0]-jv[5,0]) + l_leg_gravity[5,0]
        
        #PD
        if L_pitch <0:
            torque[4,0] = 0.1*(0-L_pitch) + 0.2
        elif L_pitch >=0:
            torque[4,0] = 0.1*(0-L_pitch) - 0.2
        if L_roll <0:
            torque[5,0] = 0.1*(0-L_roll) + 0.2
        elif L_roll >=0:
            torque[5,0] = 0.1*(0-L_roll) - 0.2
        
        # # 直接不給
        # torque[4,0] = 0
        # torque[5,0] = 0


        torque[6,0] = kr[0,0]*(vr_cmd[0,0]-jv[6,0]) + r_leg_gravity[0,0]
        torque[7,0] = kr[1,0]*(vr_cmd[1,0]-jv[7,0])+ r_leg_gravity[1,0]
        torque[8,0] = kr[2,0]*(vr_cmd[2,0]-jv[8,0]) + r_leg_gravity[2,0]
        torque[9,0] = kr[3,0]*(vr_cmd[3,0]-jv[9,0]) + r_leg_gravity[3,0]

        #運動學
        # torque[10,0] = kr[4,0]*(vr_cmd[4,0]-jv[10,0]) + r_leg_gravity[4,0]
        # torque[11,0] = kr[5,0]*(vr_cmd[5,0]-jv[11,0]) + r_leg_gravity[5,0]

        #PD
        if R_pitch <0:
            torque[10,0] = 0.1*(0-R_pitch)
        elif R_pitch >=0:
            torque[10,0] = 0.1*(0-R_pitch)
        if R_roll <0:
            torque[11,0] = 0.1*(0-R_roll)
        elif R_roll >=0:
            torque[11,0] = 0.1*(0-R_roll)

        # # 直接不給
        # torque[10,0] = 0
        # torque[11,0] = 0
       
        vcmd_data = np.array([[vl_cmd[0,0]],[vl_cmd[1,0]],[vl_cmd[2,0]],[vl_cmd[3,0]],[vl_cmd[4,0]],[vl_cmd[5,0]]])
        node.vcmd_publisher.publish(Float64MultiArray(data=vcmd_data))
        jv_collect = np.array([[jv[0,0]],[jv[1,0]],[jv[2,0]],[jv[3,0]],[jv[4,0]],[jv[5,0]]])
        node.velocity_publisher.publish(Float64MultiArray(data=jv_collect))#檢查收到的速度(超髒)

        return torque
    
    @staticmethod
    def alip_L(node,stance_type,px_in_lf,torque_ALIP,com_in_lf,state):
        # print("ALIP_L")
        stance = copy.deepcopy(stance_type) 
        #獲得kine算出來的關節扭矩 用於後續更改腳踝扭矩
        torque = copy.deepcopy(torque_ALIP) 
        com_in_wf = copy.deepcopy(node.P_COM_wf)
        lx_in_wf = copy.deepcopy(node.P_L_wf)

        #質心相對L frame的位置
        PX_l = com_in_wf - lx_in_wf
        PX_l[0,0] = PX_l[0,0] #xc
        PX_l[1,0] = PX_l[1,0] #yc

        #計算質心速度(v從世界座標下求出)
        node.CX_dot_L = (com_in_wf[0,0] - node.CX_past_L)/node.timer_period
        node.CX_past_L = com_in_wf[0,0]
        node.CY_dot_L = (com_in_wf[1,0] - node.CY_past_L)/node.timer_period
        node.CY_past_L = com_in_wf[1,0]

        #velocity filter
        node.Vx_L = 0.7408*node.Vx_past_L + 0.2592*node.CX_dot_past_L  #濾過後的速度(5Hz)
        node.Vx_past_L = node.Vx_L
        node.CX_dot_past_L =  node.CX_dot_L

        node.Vy_L = 0.7408*node.Vy_past_L + 0.2592*node.CY_dot_past_L  #濾過後的速度(5Hz)
        node.Vy_past_L = node.Vy_L
        node.CY_dot_past_L =  node.CY_dot_L

        #量測值
        Xc_mea = PX_l[0,0]
        Ly_mea = 9*node.Vx_L*0.45
        Yc_mea = PX_l[1,0]
        Lx_mea = -9*node.Vy_L*0.45 #(記得加負號)
        node.mea_x_L = np.array([[Xc_mea],[Ly_mea]])
        node.mea_y_L = np.array([[Yc_mea],[Lx_mea]])
       
        #參考值(直接拿從online_planning來的)
        # ref_x_L = copy.deepcopy(node.ref_x_L)
        # ref_y_L = copy.deepcopy(node.ref_y_L)
        
        ref_x_L = np.vstack(( 0.0, 0.0 ))
        ref_y_L = np.vstack((-0.1, 0.0))
        
        #xc & ly model(m=9 H=0.45 Ts=0.01)
        Ax = np.array([[1,0.00247],[0.8832,1]])
        Bx = np.array([[0],[0.01]])
        Cx = np.array([[1,0],[0,1]])  
        #--LQR
        # Kx = np.array([[290.3274,15.0198]])
        Kx = np.array([[150,15.0198]])
        Lx = np.array([[0.1390,0.0025],[0.8832,0.2803]])
        # Kx = np.array([[184.7274,9.9032]])
        # Lx = np.array([[0.1427,-0.0131],[0.8989,0.1427]]) 
        #--compensator
        node.ob_x_L = Ax@node.ob_x_past_L + node.ap_past_L*Bx + Lx@(node.mea_x_past_L - Cx@node.ob_x_past_L)

        #由於程式邏輯 使得左腳在擺動過程也會估測 然而並不會拿來使用
        #為了確保支撐腳切換過程 角動量估測連續性
        if node.stance_past == 0 and node.stance == 1:
            node.mea_x_L[1,0] = copy.deepcopy(node.mea_x_past_R[1,0])
            node.ob_x_L[1,0] = copy.deepcopy(node.ob_x_past_R[1,0])

        #----calculate toruqe
        # node.ap_L = -Kx@(node.ob_x_L)  #(地面給機器人 所以使用時要加負號)
        # node.ap_L = -torque[4,0] #torque[4,0]為左腳pitch對地,所以要加負號才會變成地對機器人
        
        # node.ap_L = -Kx@(node.ob_x_L - ref_x_L)*0.5
        node.ap_L = -Kx@(node.mea_x_L - ref_x_L)

        # if node.ap_L >= 3:
        #     node.ap_L = 3
        # elif node.ap_L <= -3:
        #     node.ap_L =-3

        #切換瞬間 扭矩切成0 避免腳沒踩穩
        if node.stance_past == 0 and node.stance == 1:
            node.ap_L = 0

        #--torque assign
        torque[4,0] = -node.ap_L
        #----update
        node.mea_x_past_L = node.mea_x_L
        node.ob_x_past_L = node.ob_x_L
        node.ap_past_L = node.ap_L

        #yc & lx model
        Ay = np.array([[1,-0.00247],[-0.8832,1]])
        By = np.array([[0],[0.01]])
        Cy = np.array([[1,0],[0,1]])  
        #--LQR
        # Ky = np.array([[-177.0596,9.6014]])
        Ky = np.array([[-150,15]])
        
        Ly = np.array([[0.1288,-0.0026],[-0.8832,0.1480]])
        #--compensator
        node.ob_y_L = Ay@node.ob_y_past_L + node.ar_past_L*By + Ly@(node.mea_y_past_L - Cy@node.ob_y_past_L)

        #由於程式邏輯 使得左腳在擺動過程也會估測 然而並不會拿來使用，因此踩踏瞬間角動量來自上一時刻
        #為了確保支撐腳切換過程 角動量估測連續性
        if node.stance_past == 0 and node.stance == 1:
            node.mea_y_L[1,0] = copy.deepcopy(node.mea_y_past_R[1,0])
            node.ob_y_L[1,0] = copy.deepcopy(node.ob_y_past_R[1,0])

        #----calculate toruqe
        # node.ar_L = -Ky@(node.ob_y_L)
        # node.ar_L = -torque[5,0]#torque[5,0]為左腳roll對地,所以要加負號才會變成地對機器人
        
        # node.ar_L = -Ky@(node.ob_y_L - ref_y_L)*0.15
        node.ar_L = -Ky@(node.mea_y_L - ref_y_L)

        # if node.ar_L >= 3:
        #     node.ar_L =3
        # elif node.ar_L <= -3:
        #     node.ar_L =-3

        #切換瞬間 扭矩切成0 避免腳沒踩穩
        if node.stance_past == 0 and node.stance == 1:
            node.ar_L = 0

        #--torque assign
        torque[5,0] = -node.ar_L
        # torque[5,0] = 0
        #----update
        node.mea_y_past_L = node.mea_y_L
        node.ob_y_past_L = node.ob_y_L
        node.ar_past_L = node.ar_L

        # node.effort_publisher.publish(Float64MultiArray(data=torque))
        tl_data= np.array([[torque[4,0]],[torque[5,0]]])
        node.torque_L_publisher.publish(Float64MultiArray(data=tl_data))


        if stance == 1:
            alip_x_data = np.array([[ref_x_L[0,0]],[ref_x_L[1,0]],[node.ob_x_L[0,0]],[node.ob_x_L[1,0]]])
            alip_y_data = np.array([[ref_y_L[0,0]],[ref_y_L[1,0]],[node.ob_y_L[0,0]],[node.ob_y_L[1,0]]])
            # alip_x_data = np.array([[node.ref_x_L[0,0]],[node.ref_x_L[1,0]],[node.mea_x_L[0,0]],[node.mea_x_L[1,0]]])
            # alip_y_data = np.array([[node.ref_y_L[0,0]],[node.ref_y_L[1,0]],[node.mea_y_L[0,0]],[node.mea_y_L[1,0]]])
            node.alip_x_publisher.publish(Float64MultiArray(data=alip_x_data))
            node.alip_y_publisher.publish(Float64MultiArray(data=alip_y_data))
            
            # if state == 30:
            #     collect_data = [str(ref_x_L[0,0]),str(ref_x_L[1,0]),str(node.ob_x_L[0,0]),str(node.ob_x_L[1,0]),
            #                     str(ref_y_L[0,0]),str(ref_y_L[1,0]),str(node.ob_y_L[0,0]),str(node.ob_y_L[1,0])]
            #     csv_file_name = '/home/ldsc/collect/alip_data.csv'
            #     with open(csv_file_name, 'a', newline='') as csvfile:
            #         # Create a CSV writer object
            #         csv_writer = csv.writer(csvfile)
            #         # Write the data
            #         csv_writer.writerow(collect_data)

        return torque

    @staticmethod
    def alip_R(node,stance_type,px_in_rf,torque_ALIP,com_in_rf,state):
        # print("ALIP_R")

        torque = copy.deepcopy(torque_ALIP) 
        stance = copy.deepcopy(stance_type) 
        #獲取量測值(相對於右腳腳底)
        # PX_r = copy.deepcopy(com_in_rf)
        com_in_wf = copy.deepcopy(node.P_COM_wf)
        rx_in_wf = copy.deepcopy(node.P_R_wf)
        PX_r = com_in_wf - rx_in_wf
       
        #計算質心速度
        node.CX_dot_R = (com_in_wf[0,0] - node.CX_past_R)/node.timer_period
        node.CX_past_R = com_in_wf[0,0]
        node.CY_dot_R = (com_in_wf[1,0] - node.CY_past_R)/node.timer_period
        node.CY_past_R = com_in_wf[1,0]

        #velocity filter
        node.Vx_R = 0.7408*node.Vx_past_R + 0.2592*node.CX_dot_past_R  #濾過後的速度(5Hz)
        node.Vx_past_R = node.Vx_R
        node.CX_dot_past_R =  node.CX_dot_R

        node.Vy_R = 0.7408*node.Vy_past_R + 0.2592*node.CY_dot_past_R  #濾過後的速度(5Hz)
        node.Vy_past_R = node.Vy_R
        node.CY_dot_past_R =  node.CY_dot_R

        #量測值
        Xc_mea = PX_r[0,0]
        Ly_mea = 9*node.Vx_R*0.45
        Yc_mea = PX_r[1,0]
        Lx_mea = -9*node.Vy_R*0.45 #(記得加負號)
        node.mea_x_R = np.array([[Xc_mea],[Ly_mea]])
        node.mea_y_R = np.array([[Yc_mea],[Lx_mea]])

        #參考值(直接拿從online_planning來的)
        ref_x_R = 0
        ref_y_R = 0.1
        # node.PX_ref = np.array([[0.0],[0.0],[0.57],[0.0],[0.0],[0.0]])
        # node.LX_ref = np.array([[0.0],[0.1],[0.0],[0.0],[0.0],[0.0]])
        # node.RX_ref = np.array([[0.0],[-0.1],[0.0],[0.0],[0.0],[0.0]])

        #xc & ly model(m=9 H=0.45 Ts=0.01)
        Ax = np.array([[1,0.00247],[0.8832,1]])
        Bx = np.array([[0],[0.01]])
        Cx = np.array([[1,0],[0,1]])  
        #--LQR
        Kx = np.array([[290.3274,15.0198]])
        Lx = np.array([[0.1390,0.0025],[0.8832,0.2803]]) 
        # Kx = np.array([[184.7274,9.9032]])
        # Lx = np.array([[0.1427,-0.0131],[0.8989,0.1427]]) 
       
        #--compensator
        node.ob_x_R = Ax@node.ob_x_past_R + node.ap_past_R*Bx + Lx@(node.mea_x_past_R - Cx@node.ob_x_past_R)

        #由於程式邏輯 使得右腳在擺動過程也會估測 然而並不會拿來使用
        #為了確保支撐腳切換過程 角動量估測連續性
        if node.stance_past == 1 and node.stance == 0:
            node.mea_x_R[1,0] = copy.deepcopy(node.mea_x_past_L[1,0])
            node.ob_x_R[1,0] = copy.deepcopy(node.ob_x_past_L[1,0])
        
        #----calculate toruqe
        # node.ap_R = -Kx@(node.ob_x_R)  #(地面給機器人 所以使用時要加負號)
        # node.ap_R = -torque[10,0] #torque[10,0]為右腳pitch對地,所以要加負號才會變成地對機器人
        node.ap_R = -Kx@(node.ob_x_R - ref_x_R)*0.5

        # if node.ap_R >= 3:
        #     node.ap_R =3
        # elif node.ap_R <= -3:
        #     node.ap_R =-3

        #切換瞬間 扭矩切成0 避免腳沒踩穩
        if node.stance_past == 1 and node.stance == 0:
            node.ap_R = 0
       
        #--torque assign
        torque[10,0] = -node.ap_R
        #----update
        node.mea_x_past_R = node.mea_x_R
        node.ob_x_past_R = node.ob_x_R
        node.ap_past_R = node.ap_R

        #yc & lx model
        Ay = np.array([[1,-0.00247],[-0.8832,1]])
        By = np.array([[0],[0.01]])
        Cy = np.array([[1,0],[0,1]])  
        #--LQR
        # Ky = np.array([[-290.3274,15.0198]])
        # Ly = np.array([[0.1390,-0.0025],[-0.8832,0.2803]])
        Ky = np.array([[-177.0596,9.6014]])
        Ly = np.array([[0.1288,-0.0026],[-0.8832,0.1480]])
        #--compensator
        node.ob_y_R = Ay@node.ob_y_past_R + node.ar_past_R*By + Ly@(node.mea_y_past_R - Cy@node.ob_y_past_R)

        #由於程式邏輯 使得右腳在擺動過程也會估測 然而並不會拿來使用
        #為了確保支撐腳切換過程 角動量估測連續性
        if node.stance_past == 1 and node.stance == 0:
            node.mea_y_R[1,0] = copy.deepcopy(node.mea_y_past_L[1,0])
            node.ob_y_R[1,0] = copy.deepcopy(node.ob_y_past_L[1,0])

        #----calculate toruqe
        # node.ar_R = -Ky@(node.ob_y_R)
        # node.ar_R = -torque[11,0]#torque[11,0]為右腳roll對地,所以要加負號才會變成地對機器人
        node.ar_R = -Ky@(node.ob_y_R - ref_y_R)*0.15

        #切換瞬間 扭矩切成0 避免腳沒踩穩
        if node.stance_past == 1 and node.stance == 0:
            node.ar_R = 0

        # if node.ar_R >= 3:
        #     node.ar_R =3
        # elif node.ar_R <= -3:
        #     node.ar_R =-3

        #--torque assign
        torque[11,0] = -node.ar_R
        #----update
        node.mea_y_past_R = node.mea_y_R
        node.ob_y_past_R = node.ob_y_R
        node.ar_past_R = node.ar_R


        # if stance == 0:
        #     alip_x_data = np.array([[ref_x_R[0,0]],[ref_x_R[1,0]],[node.ob_x_R[0,0]],[node.ob_x_R[1,0]]])
        #     alip_y_data = np.array([[ref_y_R[0,0]],[ref_y_R[1,0]],[node.ob_y_R[0,0]],[node.ob_y_R[1,0]]])
        #     # alip_x_data = np.array([[node.ref_x_R[0,0]],[node.ref_x_R[1,0]],[node.mea_x_R[0,0]],[node.mea_x_R[1,0]]])
        #     # alip_y_data = np.array([[node.ref_y_R[0,0]],[node.ref_y_R[1,0]],[node.mea_y_R[0,0]],[node.mea_y_R[1,0]]])
        #     node.alip_x_publisher.publish(Float64MultiArray(data=alip_x_data))
        #     node.alip_y_publisher.publish(Float64MultiArray(data=alip_y_data))
        #     # if state == 30:
        #     #     collect_data = [str(ref_x_R[0,0]),str(ref_x_R[1,0]),str(node.ob_x_R[0,0]),str(node.ob_x_R[1,0]),
        #     #                     str(ref_y_R[0,0]),str(ref_y_R[1,0]),str(node.ob_y_R[0,0]),str(node.ob_y_R[1,0])]
        #     #     csv_file_name = '/home/ldsc/collect/alip_data.csv'
        #     #     with open(csv_file_name, 'a', newline='') as csvfile:
        #     #         # Create a CSV writer object
        #     #         csv_writer = csv.writer(csvfile)
        #     #         # Write the data
        #     #         csv_writer.writerow(collect_data)
    
        return torque
    