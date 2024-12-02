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

# from linkattacher_msgs.srv import AttachLink
# from linkattacher_msgs.srv import DetachLink

#========================常數==================================#
original_joint_order = ['L_Hip_az', 'L_Hip_ay', 'L_Knee_ay', 'L_Ankle_ay', 'L_Ankle_ax', 'R_Hip_ax',
                        'R_Hip_az', 'R_Knee_ay', 'R_Hip_ay', 'R_Ankle_ay', 'L_Hip_ax', 'R_Ankle_ax']
desired_joint_order = ['L_Hip_ax', 'L_Hip_az', 'L_Hip_ay', 'L_Knee_ay', 'L_Ankle_ay', 'L_Ankle_ax',
                       'R_Hip_ax', 'R_Hip_az', 'R_Hip_ay', 'R_Knee_ay', 'R_Ankle_ay', 'R_Ankle_ax']

timer_period = 0.01 # seconds 跟joint state update rate&取樣次數有關
DS_timeLength = 2 # 雙支撐的時間總長

#理想機器人狀態
m = 9    #機器人下肢總重
H = 0.45 #理想質心高度
W = 0.2 #兩腳底間距
g = 9.81 #重力
l = sqrt(g/H)
T = 0.5 #支撐間隔時長

def make_2order_bw_iterfilter(wc):
    '''
    二階Butterworth filter的transfer function
        (a0 + a1 z^-1 + a2 z^-2)
    ----------------------------
        (b0 + b1 z^-1 + b2 z^-2)
    '''
    T=timer_period
    
    b0 = (T*wc)**2 + 8**0.5*T*wc +4
    b1 = 2*(T*wc)**2 - 8
    b2 = (T*wc)**2 - 8**0.5*T*wc + 4
    
    a0 = (T*wc)**2
    a1 = 2*(T*wc)**2
    a2 = (T*wc)**2
    
    #==================剛啟動沒有歷史值不濾波==========#
    u_pp = yield
    y_pp = u_pp
    u_p = yield y_pp
    y_p = u_p
    u = yield y_p
    
    while True:
        #=========================差分方程式==========================#
        y = (a0*u + a1*u_p + a2*u_pp - b1*y_p -b2*y_pp ) / b0
        #==================更新舊值且輸出濾波結果,暫停等到下一輪輸入==========#
        y_pp, y_p = y_p, y
        u_pp, u_p = u_p, u
        
        u = yield y

def diff2vel():
    position_p = yield
    position = yield position_p * 0
    while True:
        vel = (position-position_p)/timer_period
        position_p = position
        position = yield vel

jpfilter = make_2order_bw_iterfilter(2*pi*30) ; next(jpfilter)
jpdiff = diff2vel(); next(jpdiff)
jvfilter = make_2order_bw_iterfilter(2*pi*30) ; next(jvfilter)

comdiff = diff2vel(); next(comdiff)
vcom_filter = make_2order_bw_iterfilter(2*pi*30) ; next(vcom_filter)

def PDcontroller(kp,kd):
    '''
    PD控制器
        (a0 + a1 z^-1)
    ----------------------------
        (b0 + b1 z^-1)
    '''
    T=timer_period
    
    b0 = T
    b1 = T
    
    a0 = 2*kd+T*kp
    a1 = T*kp-2*kd
    
    #==================剛啟動沒有歷史值不濾波==========#
    
    u_p = 0
    y_p = 0
    u = yield
    
    while True:
        #=========================差分方程式==========================#
        y = (a0*u + a1*u_p - b1*y_p) / b0
        #==================更新舊值且輸出濾波結果,暫停等到下一輪輸入==========#
        y_p = y
        u_p = u
        
        u = yield y

PD_sw_x = PDcontroller(0.1,0); next(PD_sw_x)
PD_sw_y = PDcontroller(0.1,0); next(PD_sw_y)

PD_cf_x = PDcontroller(0.3,0); next(PD_cf_x)
PD_cf_y = PDcontroller(0.1,0); next(PD_cf_y)

isGotten = False
def get_initial_place(self):
    global isGotten
    if isGotten:
        pass
    else:
        self.ref_pa_pel_in_wf = self.pt.pa_pel_in_wf
        self.ref_pa_lf_in_wf  = self.pt.pa_lf_in_wf
        self.ref_pa_rf_in_wf  = self.pt.pa_rf_in_wf
        isGotten = True
        
class Allcoodinate():
    '''儲存所有座標軸的位置、速度、旋轉矩陣相對其他frame'''
    def __init__(self) -> None:
        #=========base_in_wf_callback=========#
        self.p_base_in_wf = None
        self.r_base2wf = None
        #=========ALIP_trajRef_planning=========#
        self.xLy0_com_in_cf = None #切換瞬間拿到的初始值
        self.yLx0_com_in_cf = None
        123
        self.p0_cf_in_wf = None
        self.p0_sw2com_in_cf = None
        
        #=========estimatorcontrol=========#
        # self.xLy_com_in_lf_past = np.zeros((2,1))
        # self.xLy_com_in_rf_past = np.zeros((2,1))
        # self.yLx_com_in_lf_past = np.zeros((2,1))
        # self.yLx_com_in_rf_past = np.zeros((2,1))        
    
    def get_all_frame_needed(self, config, bipedal_floating_model, bipedal_floating_data, jp, jv, cf) -> None:
        self._get_in_pf(config, bipedal_floating_model, bipedal_floating_data, jp, jv)
        self._get_in_wf() #利用座標轉換求得_in_wf
        self._jointframe_rotation(jp)
        self._get_unitVector_of_jv()
        self._get_pL_com_in_foot(jv)
            
    def _get_in_pf(self, config:pink.Configuration, bipedal_floating_model, bipedal_floating_data, jp:np.ndarray, jv:np.ndarray):
        
        def config_to_linkplace():
            '''利用self.robot的config求出各點的齊次矩陣'''
            #===========================用pinocchio求質心==================================#
            pin.centerOfMass(bipedal_floating_model, bipedal_floating_data, jp) # 會變更引數的值, 不會return
            # bipedal_floating_data.com[0] 是一維的np.array
            p_com_in_pf = np.vstack([ *bipedal_floating_data.com[0] ]) #todo 可以確認看看不同模型建立出來的質心會不會不一樣
            yield p_com_in_pf
            
            #=============================================================#
            for linkname in linknames:
                homo_pt_in_pf = config.get_transform_frame_to_world(linkname) #骨盆對pink_wf齊次矩陣
                
                p_pt_in_pf = homo_pt_in_pf.translation
                r_pt2pf = homo_pt_in_pf.rotation
                
                a_pt_in_pf = Rotation.from_matrix(r_pt2pf).as_euler('zyx', degrees=False) [::-1] #把旋轉矩陣換成歐拉角zyx,並轉成ax,ay,az
                
                pa_pt_in_pf=np.vstack(( *p_pt_in_pf, *a_pt_in_pf ))
                
                yield pa_pt_in_pf, r_pt2pf
                
        linknames = [
            "pelvis_link", "l_foot", "r_foot", 
            "l_hip_yaw_1", "l_hip_pitch_1", "l_thigh_1", "l_shank_1", "l_ankle_1", "l_foot_1",
            "r_hip_yaw_1", "r_hip_pitch_1", "r_thigh_1", "r_shank_1", "r_ankle_1", "r_foot_1"
            ]
        (
            self.p_com_in_pf,
            (self.pa_pel_in_pf, self.r_pel2pf), #骨盆
            (self.pa_lf_in_pf , self.r_lf2pf ), #左腳掌
            (self.pa_rf_in_pf , self.r_rf2pf ), #右腳掌
            
            (self.pa_lj1_in_pf, self.r_lj12pf), #由骨盆往下第一個左邊關節
            (self.pa_lj2_in_pf, self.r_lj22pf),
            (self.pa_lj3_in_pf, self.r_lj32pf),
            (self.pa_lj4_in_pf, self.r_lj42pf),
            (self.pa_lj5_in_pf, self.r_lj52pf),
            (self.pa_lj6_in_pf, self.r_lj62pf),
            
            (self.pa_rj1_in_pf, self.r_rj12pf), #由骨盆往下第一個右邊關節
            (self.pa_rj2_in_pf, self.r_rj22pf),
            (self.pa_rj3_in_pf, self.r_rj32pf),
            (self.pa_rj4_in_pf, self.r_rj42pf),
            (self.pa_rj5_in_pf, self.r_rj52pf),
            (self.pa_rj6_in_pf, self.r_rj62pf) 
        )= list(config_to_linkplace())
                     
    def _get_in_wf(self):
        def inpf_to_inwf():
            r_pf2pel = self.r_pel2pf.T
            p_pel_in_pf = self.pa_pel_in_pf[0:3]
            
            for pt in pts:
                p_pt_in_pf = self.p_com_in_pf if pt =='com' else \
                             self.__dict__[f'pa_{pt}_in_pf'][0:3]
                             
                p_pt_in_wf = self.r_pel2wf @ r_pf2pel @(p_pt_in_pf - p_pel_in_pf) + self.p_pel_in_wf
                if pt =='com':
                    yield p_pt_in_wf
                else:
                    r_pt2pf = self.__dict__[f'r_{pt}2pf']
                    r_pt2wf = self.r_pel2wf @ r_pf2pel @ r_pt2pf
                
                    if pt in ('lf','rf'):
                        #==========利用旋轉矩陣算出角度'''========#
                        a_pt_in_wf = Rotation.from_matrix(r_pt2wf).as_euler('zyx', degrees=False) [::-1] #把旋轉矩陣換成歐拉角zyx,並轉成ax,ay,az
                        yield p_pt_in_wf, r_pt2wf, np.vstack([* a_pt_in_wf])
                    else:
                        yield p_pt_in_wf, r_pt2wf
                
        #==========利用base_in_wf_callback訂閱到的base_in_wf的位態求出pel_in_wf的位態========#
        self.p_pel_in_base = np.vstack(( 0.0, 0.0, 0.598 ))
        
        self.p_pel_in_wf = self.r_base2wf @ self.p_pel_in_base + self.p_base_in_wf
        self.r_pel2wf = deepcopy(self.r_base2wf) #因為兩者是平移建立的
        a_pel_in_wf = Rotation.from_matrix(self.r_pel2wf).as_euler('zyx', degrees=False) [::-1] #把旋轉矩陣換成歐拉角zyx,並轉成ax,ay,az
        self.a_pel_in_wf = np.vstack([*a_pel_in_wf])
        
        #==========利用訂閱推導出的的pel_in_wf,算出pf下的向量,再轉到pel,再轉到wf'''========#
        pts = [
            'com', 'lf', 'rf',
            'lj1', 'lj2', 'lj3', 'lj4', 'lj5', 'lj6',
            'rj1', 'rj2', 'rj3', 'rj4', 'rj5', 'rj6'
            ]
        
        (
            self.p_com_in_wf, #質心
            (self.p_lf_in_wf , self.r_lf2wf, self.a_lf_in_wf ), #左腳掌
            (self.p_rf_in_wf , self.r_rf2wf, self.a_rf_in_wf ), #右腳掌
            
            (self.p_lj1_in_wf, self.r_lj12wf), #由骨盆往下第一個左邊關節
            (self.p_lj2_in_wf, self.r_lj22wf),
            (self.p_lj3_in_wf, self.r_lj32wf),
            (self.p_lj4_in_wf, self.r_lj42wf),
            (self.p_lj5_in_wf, self.r_lj52wf),
            (self.p_lj6_in_wf, self.r_lj62wf),
            
            (self.p_rj1_in_wf, self.r_rj12wf), #由骨盆往下第一個右邊關節
            (self.p_rj2_in_wf, self.r_rj22wf),
            (self.p_rj3_in_wf, self.r_rj32wf),
            (self.p_rj4_in_wf, self.r_rj42wf),
            (self.p_rj5_in_wf, self.r_rj52wf),
            (self.p_rj6_in_wf, self.r_rj62wf) 
            ) = list(inpf_to_inwf())
        self.pa_pel_in_wf = np.vstack(( self.p_pel_in_wf, self.a_pel_in_wf ))
        self.pa_lf_in_wf = np.vstack(( self.p_lf_in_wf, self.a_lf_in_wf ))
        self.pa_rf_in_wf = np.vstack(( self.p_rf_in_wf, self.a_rf_in_wf ))
    
    def _jointframe_rotation(self,jp):
        '''得到兩兩相鄰的關節frame彼此的關係'''
        def xyz_rotation(axis:str, theta:float) ->np.ndarray:
            '''關節繞自己軸旋轉->和前一個frame的旋轉矩陣'''
            if axis == 'x':
                return np.array([
                    [1, 0,           0],
                    [0, cos(theta), -sin(theta)],
                    [0, sin(theta),  cos(theta)]
                    ])
            if axis == 'y':
                return np.array([
                    [cos(theta), 0, sin(theta)],
                    [0,          1, 0],
                    [-sin(theta),0, cos(theta)]
                    ])
            if axis == 'z':
                return np.array([
                    [cos(theta), -sin(theta),0],
                    [sin(theta),  cos(theta),0],
                    [0,           0,         1]
                    ])
                
        def jointframe_rotation_generator():
            '''生成器, 嗯,我也不知道要怎麼命名'''
            for i,joint_name in enumerate( desired_joint_order ):
                axis = joint_name[-1]
                yield xyz_rotation(axis,jp[i,0])
                
        (
            self.r_lj12pel, self.r_lj22lj1, self.r_lj32lj2, 
            self.r_lj42lj3, self.r_lj52lj4, self.r_lj62lj5,
            
            self.r_rj12pel, self.r_rj22rj1, self.r_rj32rj2, 
            self.r_rj42rj3, self.r_rj52rj4, self.r_rj62rj5
            
            ) = list(jointframe_rotation_generator())
        
    def _get_unitVector_of_jv_in_cf(self,cf):
        '''求每個關節的旋轉軸在wf的方向->為關節角動量的方向'''

        self.u_axis_lj1_in_pf = self.r_lj12pf @ np.vstack(( 1, 0, 0 )) #最後面的向量是指旋轉軸向量in自己的frame
        self.u_axis_lj2_in_pf = self.r_lj22pf @ np.vstack(( 0, 0, 1 ))
        self.u_axis_lj3_in_pf = self.r_lj32pf @ np.vstack(( 0, 1, 0 ))
        self.u_axis_lj4_in_pf = self.r_lj42pf @ np.vstack(( 0, 1, 0 ))
        self.u_axis_lj5_in_pf = self.r_lj52pf @ np.vstack(( 0, 1, 0 ))
        self.u_axis_lj6_in_pf = self.r_lj62pf @ np.vstack(( 1, 0, 0 ))
        
        self.u_axis_rj1_in_pf = self.r_rj12pf @ np.vstack(( 1, 0, 0 ))
        self.u_axis_rj2_in_pf = self.r_rj22pf @ np.vstack(( 0, 0, 1 ))
        self.u_axis_rj3_in_pf = self.r_rj32pf @ np.vstack(( 0, 1, 0 ))
        self.u_axis_rj4_in_pf = self.r_rj42pf @ np.vstack(( 0, 1, 0 ))
        self.u_axis_rj5_in_pf = self.r_rj52pf @ np.vstack(( 0, 1, 0 ))
        self.u_axis_rj6_in_pf = self.r_rj62pf @ np.vstack(( 1, 0, 0 ))
        
        r_pf2lf = self.r_lf2pf.T
        r_pf2rf = self.r_rf2pf.T
        self.u_axis_lj1_in_lf = r_pf2lf @ self.u_axis_lj1_in_pf
        self.u_axis_lj2_in_lf = r_pf2lf @ self.u_axis_lj2_in_pf
        self.u_axis_lj3_in_lf = r_pf2lf @ self.u_axis_lj3_in_pf
        self.u_axis_lj4_in_lf = r_pf2lf @ self.u_axis_lj4_in_pf
        self.u_axis_lj5_in_lf = r_pf2lf @ self.u_axis_lj5_in_pf
        self.u_axis_lj6_in_lf = r_pf2lf @ self.u_axis_lj6_in_pf
        
        self.u_axis_rj1_in_rf = r_pf2rf @ self.u_axis_rj1_in_pf
        self.u_axis_rj2_in_rf = r_pf2rf @ self.u_axis_rj2_in_pf
        self.u_axis_rj3_in_rf = r_pf2rf @ self.u_axis_rj3_in_pf
        self.u_axis_rj4_in_rf = r_pf2rf @ self.u_axis_rj4_in_pf
        self.u_axis_rj5_in_rf = r_pf2rf @ self.u_axis_rj5_in_pf
        self.u_axis_rj6_in_rf = r_pf2rf @ self.u_axis_rj6_in_pf
        
        
    def _get_pL_com_in_foot(self, jv):
        """ 用wf的相對關係求出com_in_ft, 再差分得到角動量, 在ALIP軌跡規劃的切換瞬間用到"""
        p_com_in_lf = self.r_lf2wf.T @ (self.p_com_in_wf - self.p_lf_in_wf)
        p_com_in_rf = self.r_rf2wf.T @ (self.p_com_in_wf - self.p_rf_in_wf)
        p_com_in_lfrf = np.vstack(( p_com_in_lf, p_com_in_rf ))
        v_com_in_lfrf = vcom_filter.send(comdiff.send(p_com_in_lfrf))
        
        J_lj2lf_in_wf, J_rj2rf_in_wf = tuple(self.Jacobian())
        v_pel_in_lf = J_lj2lf_in_wf @ jv[:6]
        v_com_in_lf = v_pel_in_lf #先用骨盆的速度代替
        
        
        lf2pel_in_wf = (self.p_pel_in_wf - self.p_lf_in_wf)
        rf2pel_in_wf = (self.p_pel_in_wf - self.p_rf_in_wf)
        Ly_com_in_lf = m * v_com_in_lf[0,0] * lf2pel_in_wf[2,0]
        Ly_com_in_rf = m * v_com_in_lfrf[3,0] * rf2pel_in_wf[2,0]
        Lx_com_in_lf = -m * v_com_in_lf[1,0] * lf2pel_in_wf[2,0]
        Lx_com_in_rf = -m * v_com_in_lfrf[4,0] * rf2pel_in_wf[2,0]
        
        # Ly_com_in_lf = m * v_com_in_lfrf[0,0] * H
        # Ly_com_in_rf = m * v_com_in_lfrf[3,0] * H
        # Lx_com_in_lf = -m * v_com_in_lfrf[1,0] * H
        # Lx_com_in_rf = -m * v_com_in_lfrf[4,0] * H
        
        self.xLy_com_in_lf = np.vstack(( p_com_in_lfrf[0,0], Ly_com_in_lf )) #輸出
        self.yLx_com_in_lf = np.vstack(( p_com_in_lfrf[1,0], Lx_com_in_lf ))
        self.xLy_com_in_rf = np.vstack(( p_com_in_lfrf[3,0], Ly_com_in_rf ))
        self.yLx_com_in_rf = np.vstack(( p_com_in_lfrf[4,0], Lx_com_in_rf ))
        
    
    def Jacobian(self):
        '''算出左關節對左腳掌, 右關節對右腳掌的Jacobian'''
        #===========================左腳位置和方向的Jacobian==================================#
        for ft in ['l','r']:
            J_jnt2ft_in_wf = np.array([ [], [], [], [], [], [] ]) #先設6*1全空的一個向量
            
            for i in range(1,6+1): 
                u_axis_ftjnti_in_wf = self.__dict__[f"u_axis_{ft}j{i}_in_wf"] #關節omega的方向
                p_ftjnti_in_wf = self.__dict__[f"p_{ft}j{i}_in_wf"]
                p_ft_in_wf = self.__dict__[f"p_{ft}f_in_wf"]
                
                p_ftjnti_to_ft_in_wf = p_ft_in_wf - p_ftjnti_in_wf #力臂方向
                
                Jpi = np.cross(u_axis_ftjnti_in_wf, p_ftjnti_to_ft_in_wf, axis=0 ) # omega x r
                Jai = np.vstack(u_axis_ftjnti_in_wf)
                Ji = np.vstack(( Jpi, Jai ))
                J_jnt2ft_in_wf = np.hstack(( J_jnt2ft_in_wf,Ji ))
                
            yield J_jnt2ft_in_wf #先左再右輸出

    def zyx_analy_2_gmtry_w(self, ay:float, az:float) -> np.ndarray:
        '''我們的歐拉角是以zyx做定義, 這個函式把解析角速度轉成幾何角速度'''
        return np.array([[cos(ay)*cos(az), -sin(az), 0],
                            [cos(ay)*sin(az),  cos(az), 0],
                            [-sin(ay),         0,       1]])
            
class UpperLevelController(Node):
    def __init__(self):
                 
        def publisher_create():
            '''effort publisher是ROS2-control的力矩, 負責控制各個關節的力矩->我們程式的目的就是為了pub他'''
            self.publisher['position'] = self.create_publisher(Float64MultiArray , '/position_controller/commands', 10)
            self.publisher['velocity'] = self.create_publisher(Float64MultiArray , '/velocity_controller/commands', 10)
            self.publisher['effort'] = self.create_publisher(Float64MultiArray , '/effort_controllers/commands', 10) #這個才是重點
            self.publisher['velocity_cmd'] = self.create_publisher(Float64MultiArray , '/velocity_command/commands', 10)
            self.publisher['l_gravity'] = self.create_publisher(Float64MultiArray , '/l_gravity', 10)
            self.publisher['r_gravity'] = self.create_publisher(Float64MultiArray , '/r_gravity', 10)
            self.publisher['alip_x'] = self.create_publisher(Float64MultiArray , '/alip_x_data', 10)
            self.publisher['alip_y'] = self.create_publisher(Float64MultiArray , '/alip_y_data', 10)
            self.publisher['torque_l'] = self.create_publisher(Float64MultiArray , '/torqueL_data', 10)
            self.publisher['torque_r'] = self.create_publisher(Float64MultiArray , '/torqueR_data', 10)
            self.publisher['ref'] = self.create_publisher(Float64MultiArray , '/ref_data', 10)
            self.publisher['joint_trajectory_controller'] = self.create_publisher(JointTrajectory , '/joint_trajectory_controller/joint_trajectory', 10)
            self.publisher['pel'] = self.create_publisher(Float64MultiArray , '/px_data', 10)
            self.publisher['com'] = self.create_publisher(Float64MultiArray , '/com_data', 10)
            self.publisher['lf'] = self.create_publisher(Float64MultiArray , '/lx_data', 10)
            self.publisher['rf'] = self.create_publisher(Float64MultiArray , '/rx_data', 10)
           
        def subscriber_create():
            callcount=0 #每5次run一次 main_callback,做decimate(down sampling)降低振盪
            #========================存放訂閱的資料,每5次輸出給self==================================#
            sub = {
                'p_base_in_wf': None,
                'r_base2wf': None,
                'state': None,
                'jp': None
                }
            def base_in_wf_callback(msg):
                ''' 訂閱base_in_wf的位置和旋轉矩陣'''
                base = msg.pose.pose.position
                quaters_base = msg.pose.pose.orientation ##四元數法
                quaters_base = Rotation.from_quat([quaters_base.x, quaters_base.y, quaters_base.z, quaters_base.w])
                sub['p_base_in_wf'] = np.vstack(( base.x, base.y, base.z ))
                sub['r_base2wf'] = quaters_base.as_matrix()
                # print('base\n',sub['p_base_in_wf'].flatten())
            
            def state_callback(msg):
                """ 接收我們手動pub出的state """
                sub['state'] = msg.data[0]
                    
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
                       
            self.subscriber['base'] = self.create_subscription(Odometry, '/odom', base_in_wf_callback, 10) #base_state_subscribe
            self.subscriber['state'] = self.create_subscription(Float64MultiArray, 'state_topic', state_callback, 10)
            self.subscriber['joint_states'] = self.create_subscription(JointState, '/joint_states', joint_states_callback, 10) #joint_state_subscribe
 
        def load_URDF():
            '''導入5個模型和1個有運動學和動力學接口的robot'''
            pinocchio_model_dir = "/home/ldsc/ros2_ws/src/bipedal_floating_description/urdf/"
            
            robot = pin.RobotWrapper.BuildFromURDF( #提供完整的運動學和動力學接口
                filename = pinocchio_model_dir + "bipedal_floating.pin.urdf",
                package_dirs = ["."],
                # root_joint=pin.JointModelFreeFlyer(),
                root_joint = None,
                )
            yield robot
            
            print(f"URDF description successfully loaded in {robot}")
            
            urdf_filenames = ['bipedal_floating', 'stance_l', 'stance_r_gravity', 'bipedal_l_gravity', 'bipedal_r_gravity']
            #分別是            從骨盆建模,           左單支撐一隻腳, 右單支撐一隻腳,      雙足模型_以左腳建起,    雙足模型_以右腳建起
            
            for urdf_filename in urdf_filenames:
                filedir = pinocchio_model_dir + urdf_filename + '.xacro'
                
                model = pin.buildModelFromUrdf(filedir) #Load urdf模型，只生成模型,不包關高階的數據和功能
                print("model name: " + model.name )
                data = model.createData() # Create data required by the algorithms
                yield model, data
        
        #========================初始化一些重要的property==================================#
        self.publisher = {} # 存放create的publisher
        self.subscriber = {} # 存放create的subscriber
        self.pt = Allcoodinate() #放各個frame
        
        self.jp = None #現在的關節角度數組
        self.jv = None #現在的關節速度數組(不乾淨)
        
        self.state = 0 #放我們pub的mode
        self.isContact = {'lf':True, 'rf':True} #左右腳是否接觸地面
        
        self.cf = '2f' #放支撐腳： 左腳'lf', 右腳'rf', 雙支撐'2f'
        self.sw = None #放擺動腳: 左腳'lf', 右腳'rf', 雙支撐''(雙支撐狀態應該不會用到)
        
        self.DS_time = 0 #雙支撐的時間進度
        self.contact_t = 0.0 #接觸時間進度
        
        self.ref_pa_pel_in_wf = None #三個控制的參考軌跡
        self.ref_pa_lf_in_wf  = None
        self.ref_pa_rf_in_wf  = None
        self.ref_xy_sw2com_in_cf_T = None #切換瞬間計算出的下一步
        
        self.ref_pa_com_in_lf = None #用於估測的質心軌跡？
        self.ref_pa_com_in_rf = None
        self.ref_xLy_com_in_lf = np.zeros((2,1))
        self.ref_xLy_com_in_rf = np.zeros((2,1))
        self.ref_yLx_com_in_lf = np.zeros((2,1))
        self.ref_yLx_com_in_rf = np.zeros((2,1))
        
        self.ob_xLy_com_in_lf_past = np.zeros((2,1)) #過去估測的質心狀態
        self.ob_xLy_com_in_rf_past = np.zeros((2,1))
        self.ob_yLx_com_in_lf_past = np.zeros((2,1))
        self.ob_yLx_com_in_rf_past = np.zeros((2,1))
        
        self.ux_lf_past = np.zeros((1,1)) #過去估測的輸入力矩
        self.ux_rf_past = np.zeros((1,1))
        self.uy_lf_past = np.zeros((1,1))
        self.uy_rf_past = np.zeros((1,1))
        #==========================================================#
        
        super().__init__('upper_level_controllers') #創建了一個叫做'upper_level_controllers'的節點        
        publisher_create()
        subscriber_create()

        (
            self.robot,
            (self.bipedal_floating_model, self.bipedal_floating_data),
            (self.stance_l_model,         self.stance_l_data),
            (self.stance_r_model,         self.stance_r_data),
            (self.bipedal_l_model,        self.bipedal_l_data),
            (self.bipedal_r_model,        self.bipedal_r_data),
        ) = list(load_URDF())
        
        # Initialize meschcat visualizer
        self.viz = pin.visualize.MeshcatVisualizer(self.robot.model, self.robot.collision_model, self.robot.visual_model)
        self.robot.setVisualizer(self.viz, init=False)
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel()
        

        
        # Set initial robot configuration
        print(self.robot.model)
        print(self.robot.q0)
        self.init_configuration = pink.Configuration(self.robot.model, self.robot.data, self.robot.q0)
        self.viz.display(self.init_configuration.q)
        
        # self.attach_link_client = self.create_client(AttachLink, '/ATTACHLINK') #會把腳底frame跟大地frame在當下的距離固定住
        # self.detach_link_client = self.create_client(DetachLink, '/DETACHLINK')

    def main_callback(self):
        # self.state = 2
        def judge_step_firmly():
            '''當腳掌高度(z)在wf<0.01當作踩穩，之後可能要改'''
            self.isContact['lf'] = (self.pt.p_lf_in_wf[2,0] <= 0.01)
            self.isContact['rf'] = (self.pt.p_rf_in_wf[2,0] <= 0.01)
            print(self.pt.p_lf_in_wf[2,0], self.pt.p_rf_in_wf[2,0])
                
        def judge_stance(pa_lf2pel_in_pf:np.ndarray,  pa_rf2pel_in_pf:np.ndarray ):
            """決定不同狀態時支撐模式怎麼做切換"""
            #========================當state0時, 骨盆距離腳的側向(y)在0.06以內, 決定支撐腳==================================#
            if self.state == 0: #判斷單雙支撐
                if abs(pa_lf2pel_in_pf[1,0])<=0.06:
                    self.cf = 'lf'
                elif abs(pa_rf2pel_in_pf[1,0])<=0.06:
                    self.cf = 'rf' #單支撐
                else:
                    self.cf = '2f' #雙支撐
            #===========================================================================#
            if self.state == 2: #改寫的
                    self.cf = 'lf'
            #========================當state1時, 一開始維持雙支撐, 兩秒後換左腳單支撐==================================#
            elif self.state == 1:
                if self.DS_time <= DS_timeLength:
                    self.cf = '2f' #開始雙支撐
                    self.DS_time += timer_period #更新時間
                    print("DS",self.DS_time)
                else:
                    self.DS_time = 10.1
                    self.cf = 'lf'
            
            #========================當state30時, #地面踩夠久才換支撐腳==================================#
            elif self.state == 30:
                if abs( self.contact_t-0.5 )<=0.005:
                    self.cf, self.sw = self.sw, self.cf
            
            #========================擺動腳==================================#
            self.sw =   'rf' if self.cf == 'lf' else\
                        'lf' if self.cf == 'rf' else\
                        ''

        #========================建立現在的模型==================================#
        config = pink.Configuration(self.robot.model, self.robot.data, self.jp) 
        self.viz.display(config.q)

        #========================得到各個frame座標==================================#
        self.pt.get_all_frame_needed(config, self.bipedal_floating_model, self.bipedal_floating_data, self.jp, self.jv, self.cf)
        
        #得到相對姿勢
        pa_lf2pel_in_pf = self.pt.pa_pel_in_pf - self.pt.pa_lf_in_pf #從pink拿骨盆座標下相對於左右腳掌的骨盆位置
        pa_rf2pel_in_pf = self.pt.pa_pel_in_pf - self.pt.pa_rf_in_pf

        #===========================================================
        judge_step_firmly()#判斷是否踩穩
        # if not all(self.isContact.values()):
        #     print('no contact')
        #     self.publisher['effort'].publish( Float64MultiArray( data = np.zeros((12,1)) ))
        #     return
        judge_stance(pa_lf2pel_in_pf, pa_rf2pel_in_pf) #怎麼切支撐狀態要改!!!!!
        #========================軌跡規劃==================================#
        self.trajRef_planning()
        #========================得到內環關節速度命令==================================#
        cmd_jv= self.outerloop()
        torq = self.innerloop(cmd_jv, pa_lf2pel_in_pf, pa_rf2pel_in_pf)#相對姿勢只有state 0會用到
        
        if self.state == 1:
            self.estimatorcontrol()
        if self.state == 2:#雙支撐的重力矩
            # torq[self.sw][4:6] = self.ankle_control()
            torq[self.cf][4:6] = self.estimatorcontrol()[self.cf]
            # torq[self.cf][4:6] = self.estimatorcontrol()
            # torq[self.sw][4:6] = np.zeros((2,1))
            # torq[self.cf][4:6] = np.zeros((2,1))
        if self.state == 30:
            torq[self.sw][4:6] = self.ankle_control()[self.sw]
            torq[self.cf][4:6] = self.estimatorcontrol()[self.cf]
        
        
        write_measure_L(self)
        writetorq(self,torq)
        write_jp(self)
        # write_jv(self)
        #========================pub出扭矩命令來控制==================================#
        self.publisher['effort'].publish( Float64MultiArray( data = np.vstack((torq['lf'], torq['rf'])) ))
        printdict  = {
            'cf': self.cf,
            'sw': self.sw,
            'state':self.state,
            'contact':self.isContact
        }
        print(printdict)
        self.state_past = deepcopy(self.state)
        self.cf_past = deepcopy(self.cf)
        
        

   
    def trajRef_planning(self):
        
        def ALIP_trajRef_planning():
            """ 用ALIP規劃出cf,sw,pel _in_wf的軌跡"""
            
            def ALIP_MAT(axis:str, t:float) -> np.ndarray:
                """理想ALIP動態矩陣"""
                if axis == 'x':
                    return np.array([[ cosh(l*t),       sinh(l*t)/(m*H*l) ], 
                                    [ m*H*l*sinh(l*t),  cosh(l*t) ]])
                elif axis == 'y':
                    return np.array([[ cosh(l*t),       -sinh(l*t)/(m*H*l) ],
                                    [ -m*H*l*sinh(l*t), cosh(l*t) ]])
            
            def getNextStep_xy_sw2com_in_cf(T:float) -> np.ndarray:
                '''得到下一步擺動腳的落地點'''
                Ly_com_in_cf_T = ( ALIP_MAT('x',T) @ self.pt.xLy0_com_in_cf[0:2] )[1,0] #下步換腳瞬間的角動量
                Lx_com_in_cf_T = ( ALIP_MAT('y',T) @ self.pt.yLx0_com_in_cf[0:2] )[1,0]
                
                
                des_vx_com_in_cf_2T = 0.15 #下下步換腳瞬間的理想Vx
                des_Ly_com_in_cf_2T = m*des_vx_com_in_cf_2T*H #下下步換腳瞬間的理想Ly
                des_Lx_com_in_cf_2T = (0.5*m*H*W)*(l*sinh(l*T))/(1+cosh(l*T)) #下下步換腳瞬間的理想Lx
                if self.sw == 'rf': #下步支撐腳是rf的話要負的
                    des_Lx_com_in_cf_2T = - des_Lx_com_in_cf_2T
                    
                x_sw2com_in_cf_T = (des_Ly_com_in_cf_2T - cosh(l*T) * Ly_com_in_cf_T) / (m*H*l*sinh(l*T)) #下步換腳瞬間的位置
                y_sw2com_in_cf_T = (des_Lx_com_in_cf_2T - cosh(l*T) * Lx_com_in_cf_T) /-(m*H*l*sinh(l*T))
                return np.vstack(( x_sw2com_in_cf_T, y_sw2com_in_cf_T ))
                        
            def get_com_trajpt_in_cf(t:float) -> np.ndarray :
                '''得出到換腳前的com_in_cf的軌跡點'''
                ref_xLy_com_in_cf =  ALIP_MAT('x',t) @ self.pt.xLy0_com_in_cf
                ref_yLx_com_in_cf =  ALIP_MAT('y',t) @ self.pt.yLx0_com_in_cf
                ref_p_com_in_cf = np.vstack((ref_xLy_com_in_cf[0,0], ref_yLx_com_in_cf[0,0], H ))
                return ref_p_com_in_cf, ref_xLy_com_in_cf, ref_yLx_com_in_cf

            def get_com2sw_trajpt_in_cf(t:float, p0_sw2com_in_cf:np.ndarray, ref_xy_sw2com_in_cf_T:np.ndarray) -> np.ndarray:
                '''給初始點和下一點, 用弦波連成換腳前的sw_in_cf的軌跡點'''
                tn = t/T
                zCL = 0.02 #踏步高度
                ref_p_sw2com_in_cf = np.vstack((
                    0.5 * ( (1+cos(pi*tn)) * p0_sw2com_in_cf[0:2] + (1-cos(pi*tn)) * ref_xy_sw2com_in_cf_T ),
                    4*zCL*(tn-0.5)**2 + (H-zCL)
                ))
                return - ref_p_sw2com_in_cf #com2sw
            
           #=====================================================================#
            p_cf_in_wf = self.pt.__dict__[f"p_{self.cf}_in_wf"]
            p_sw_in_wf = self.pt.__dict__[f"p_{self.sw}_in_wf"]
            
            r_cf2wf = np.identity(3)
            r_wf2cf = r_cf2wf.T
            #====================拿取初始值,一步還沒走完前不能被改變====================#
            if self.state != self.state_past or self.cf != self.cf_past: #切換的瞬間,時間設成0,拿取初始值
                self.contact_t = 0.0 #接觸時間歸零
                self.pt.xLy0_com_in_cf = deepcopy( self.pt.__dict__[f"xLy_com_in_{self.cf}"] ) 
                self.pt.yLx0_com_in_cf = deepcopy( self.pt.__dict__[f"yLx_com_in_{self.cf}"] )
                self.pt.p0_cf_in_wf = deepcopy(self.pt.__dict__[f"p_{self.cf}_in_wf"])
                self.pt.p0_sw2com_in_cf = r_wf2cf @ (self.pt.p_com_in_wf - p_sw_in_wf) 
                
                self.ref_xy_sw2com_in_cf_T = getNextStep_xy_sw2com_in_cf(T) #預測擺動腳下一步
                
            #=====================================================================#
            
            ref_p_com_in_cf, ref_xLy_com_in_cf, ref_yLx_com_in_cf = get_com_trajpt_in_cf(self.contact_t) #得到cf下的軌跡點
            ref_p_com2sw_in_cf = get_com2sw_trajpt_in_cf(self.contact_t, self.pt.p0_sw2com_in_cf, self.ref_xy_sw2com_in_cf_T)
            
            ref_p_sw_in_wf = r_cf2wf @ ( ref_p_com_in_cf + ref_p_com2sw_in_cf ) + self.pt.p0_cf_in_wf
            # ref_p_sw_in_wf = r_cf2wf @ ref_p_com2sw_in_cf + self.pt.p_com_in_wf #sw參考軌跡
            
            ref_p_com_in_wf = r_cf2wf @ ref_p_com_in_cf + self.pt.p0_cf_in_wf #com參考軌跡
            p_com2pel_in_wf = self.pt.p_pel_in_wf - self.pt.p_com_in_wf #com和pel的相對位置
            ref_p_pel_in_wf = p_com2pel_in_wf + ref_p_com_in_wf #pel參考軌跡
            ref_p_pel_in_wf[2] = 0.55
            
            #===========================主要的參考軌跡==================================#
            ref_pa_pel_in_wf = np.vstack(( ref_p_pel_in_wf, 0, 0, 0 ))
            ref_pa_cf_in_wf = np.vstack(( self.pt.p0_cf_in_wf, 0, 0, 0 ))
            ref_pa_sw_in_wf = np.vstack(( ref_p_sw_in_wf, 0, 0, 0 ))
            
            ref_pa_com_in_cf = np.vstack(( ref_p_com_in_cf, 0, 0, 0 ))
            
            return ( ref_pa_pel_in_wf, ref_pa_cf_in_wf, ref_pa_sw_in_wf,ref_p_com_in_wf, #迴圈控制的三道軌跡
                     ref_pa_com_in_cf, ref_xLy_com_in_cf, ref_yLx_com_in_cf ) #補償控制的com參數

        if self.state == 0:
            self.ref_pa_pel_in_wf = np.vstack(( 0.0,  0.0, 0.55, 0.0, 0.0, 0.0 ))
            self.ref_pa_lf_in_wf  = np.vstack(( 0.0,  0.1, 0.0,  0.0, 0.0, 0.0 ))
            self.ref_pa_rf_in_wf  = np.vstack(( 0.0, -0.1, 0.0,  0.0, 0.0, 0.0 ))
            
        if self.state == 2:
            # self.ref_pa_pel_in_wf = np.vstack(( 0.0,  0.0, 0.56, 0.0, 0.0, 0.0 ))
            # self.ref_pa_lf_in_wf  = np.vstack(( 0.0,  0.1, 0.0,  0.0, 0.0, 0.0 ))
            # self.ref_pa_rf_in_wf  = np.vstack(( 0.0, -0.1, 0.0,  0.0, 0.0, 0.0 ))
            get_initial_place(self)
            
    
        elif self.state == 1: #左右腳都放在0.1, state1:骨盆在0.5DDT內移到左邊0.06(左腳)
            targetstate1_y_pel_in_wf = 0.06

            if self.DS_time > 0.0 and self.DS_time <= 0.5*DS_timeLength:
                ref_y_pel_in_wf = targetstate1_y_pel_in_wf * self.DS_time/ (0.5*DS_timeLength)
            else:
                ref_y_pel_in_wf = targetstate1_y_pel_in_wf

            self.ref_pa_pel_in_wf = np.vstack(( 0.0,  ref_y_pel_in_wf, 0.55, 0.0, 0.0, 0.0 ))
            self.ref_pa_lf_in_wf  = np.vstack(( 0.0,  0.1,             0.0,  0.0, 0.0, 0.0 ))
            self.ref_pa_rf_in_wf  = np.vstack(( 0.0, -0.1,             0.0,  0.0, 0.0, 0.0 ))
        
        elif self.state == 30: #ALIP模式
            (
                self.ref_pa_pel_in_wf, #迴圈控制的三道軌跡
                self.__dict__[f"ref_pa_{self.cf}_in_wf"],
                self.__dict__[f"ref_pa_{self.sw}_in_wf"],
                self.ref_p_com_in_wf,
            
                self.__dict__[f"ref_pa_com_in_{self.cf}"], #補償控制的com參數
                self.__dict__[f"ref_xLy_com_in_{self.cf}"],
                self.__dict__[f"ref_yLx_com_in_{self.cf}"],
                
            ) = ALIP_trajRef_planning() #即時規劃出參考命令(下一個軌跡點)
            
            self.contact_t += timer_period

    def outerloop(self):        

        def endErr_to_endVeocity() -> dict[str, np.ndarray]:
            '''計算對骨盆的支撐腳、擺動腳in wf的error, 並經過Pcontrol再轉成幾何速度輸出'''
            #p control #todo擺動腳之後要改成PI
            #===========================得到參考命令和量測值==================================#
            ref_pa_pel2lf_in_wf = self.ref_pa_lf_in_wf - self.ref_pa_pel_in_wf
            ref_pa_pel2rf_in_wf = self.ref_pa_rf_in_wf - self.ref_pa_pel_in_wf
            pa_pel2lf_in_wf = self.pt.pa_lf_in_wf - self.pt.pa_pel_in_wf
            pa_pel2rf_in_wf = self.pt.pa_rf_in_wf - self.pt.pa_pel_in_wf
            
            #===========================經過加法器計算error==================================#
            err_pa_pel2lf_in_wf = ref_pa_pel2lf_in_wf - pa_pel2lf_in_wf
            err_pa_pel2rf_in_wf = ref_pa_pel2rf_in_wf - pa_pel2rf_in_wf
            
            print('ref_pa_pel2lf:',ref_pa_pel2lf_in_wf.flatten())
            print('pa_pel2lf_in_wf:',pa_pel2lf_in_wf.flatten())
            print('err_pa_pel2lf_in_wf:',err_pa_pel2lf_in_wf.flatten())
            print('err_pa_pel2rf_in_wf:',err_pa_pel2rf_in_wf.flatten())

            #===========================經過kp==================================#
            derr_pa_pel2lf_in_wf = np.vstack(( 20*err_pa_pel2lf_in_wf[:3], 20*err_pa_pel2lf_in_wf[3:] ))#25 rad/s
            derr_pa_pel2rf_in_wf = np.vstack(( 20*err_pa_pel2rf_in_wf[:3], 20*err_pa_pel2rf_in_wf[3:] ))#12.5 rad/s
            
            #===========================轉換成geometry端末角速度==================================#
            w_pel2lf_in_wf = self.pt.zyx_analy_2_gmtry_w(*ref_pa_pel2lf_in_wf[-2:,0]) @ derr_pa_pel2lf_in_wf[-3:] #analytical轉成Geometry
            w_pel2rf_in_wf = self.pt.zyx_analy_2_gmtry_w(*ref_pa_pel2rf_in_wf[-2:,0]) @ derr_pa_pel2rf_in_wf[-3:]

            vw_pel2lf_in_wf = np.vstack(( derr_pa_pel2lf_in_wf[:3], w_pel2lf_in_wf )) 
            vw_pel2rf_in_wf = np.vstack(( derr_pa_pel2rf_in_wf[:3], w_pel2rf_in_wf ))
            
            return {
                'lf' : vw_pel2lf_in_wf,
                'rf' : vw_pel2rf_in_wf
            }
        
        def endVel_to_jv(vw_pel2ft_in_wf : dict[str, np.ndarray] ) -> dict[str, np.ndarray]:
            '''轉成支撐腳、擺動腳膝上4關節速度命令'''
            J_lj2lf_in_wf, J_rj2rf_in_wf = tuple(self.pt.Jacobian())
            J = {'lf': J_lj2lf_in_wf,
                 'rf': J_rj2rf_in_wf}
            jv_ankle_of = {'lf':self.jv[4:6], #左右腳踝關節轉速
                           'rf':self.jv[10:]}
            
            if self.state in [30, 2]: #ALIP mode
                
                #===========================支撐腳膝上四關節: 控骨盆z, axyz==================================#
                vzwxyz_pel2cf_in_wf = vw_pel2ft_in_wf[self.cf][2:]
                J_ankle_to_vzwxyz_cf = J[self.cf][2:, 4:]
                J_knee_to_vzwxyz_cf = J[self.cf][2:, :4]
                
                vzwxyz_indep_pel2cf_in_wf = vzwxyz_pel2cf_in_wf - J_ankle_to_vzwxyz_cf @ jv_ankle_of[self.cf] # 經過加法器扣除掉腳踝關節(i.e.膝下關節)的影響
                
                J_vzwxyz_cf_inv = np.linalg.pinv( J_knee_to_vzwxyz_cf ) # 經過J^-1
                cmd_v_j1234_cf = J_vzwxyz_cf_inv @ vzwxyz_indep_pel2cf_in_wf
                
                #===========================擺動腳膝上四關節: 控落點xyz, az==================================#
                vxyzwz_pel2sw_in_wf = np.vstack([ vw_pel2ft_in_wf[self.sw][i] for i in (0,1,2,-1) ])
                J_ankle_to_vxyzwz_sw = np.vstack([ J[self.sw][i, 4:] for i in (0,1,2,-1) ])
                J_knee_to_vxyzwz_sw = np.vstack([ J[self.sw][i, :4] for i in (0,1,2,-1) ])

                vxyzwz_indep_pel2sw_in_wf = vxyzwz_pel2sw_in_wf - J_ankle_to_vxyzwz_sw @ jv_ankle_of[self.sw] # 經過加法器扣除掉腳踝關節(i.e.膝下關節)的影響
                
                J_vxy_wz_inv = np.linalg.pinv( J_knee_to_vxyzwz_sw ) # 經過J^-1
                cmd_v_j1234_sw = J_vxy_wz_inv @ vxyzwz_indep_pel2sw_in_wf
                
                cmd_v_j1234_cf,cmd_v_j1234_sw = np.maximum( (cmd_v_j1234_cf,cmd_v_j1234_sw), -0.75 )
                cmd_v_j1234_cf,cmd_v_j1234_sw = np.minimum( (cmd_v_j1234_cf,cmd_v_j1234_sw),  0.75 )
                
                # print(J['lf'])
                # print(J_ankle_to_vzwxyz_cf)
                # print(jv_ankle_of[self.cf])
                # print(J_knee_to_vzwxyz_cf)
                print('vzwxyz_pel2cf_in_wf:',vzwxyz_pel2cf_in_wf.flatten())
                print('vzwxyz_indep_pel2cf_in_wf:',vzwxyz_indep_pel2cf_in_wf.flatten())
                
                return {
                    self.cf : np.vstack(( cmd_v_j1234_cf, 0, 0 )),
                    self.sw : np.vstack(( cmd_v_j1234_sw, 0, 0 ))
                }
            else:
                return { #不是ALIP就不用排除腳踝
                    'lf' : np.linalg.pinv(J['lf']) @ vw_pel2ft_in_wf['lf'],
                    'rf' : np.linalg.pinv(J['rf']) @ vw_pel2ft_in_wf['rf']
                }
                
                
        vw_pel2ft_in_wf = endErr_to_endVeocity()
        cmd_jv = endVel_to_jv(vw_pel2ft_in_wf)
        print('cmd_jv:',{key: cmd_jv[key].flatten()*180/pi for key in cmd_jv.keys()})
        write_jv(self,cmd_jv)
        return cmd_jv
    
    def innerloop(self, cmd_jv, pa_lf2pel_in_pf, pa_rf2pel_in_pf) -> dict[str, np.ndarray]:
        def cal_jvErr_to_ja():
            '''通過加法器計算誤差再進入kp'''
            #===========================膝上四關節進入加法器算誤差==================================#
            err_jv = {
                'lf' : cmd_jv['lf'] - self.jv[:6],
                'rf' : cmd_jv['rf'] - self.jv[6:]
            }
            #==================經過膝上四關節的kp==================#
            if self.state == 30: #腳踝關節不在這控, 設成0
                ksw = np.vstack(( 1, 1, 1, 1, 0, 0))
                kcf = np.vstack(( 1.2, 1.2, 1.2, 1.5, 0, 0)) if self.isContact[self.cf] else\
                    np.vstack(( 1.2, 1.2, 1.2, 1.2, 0, 0))
                return {
                    self.cf : kcf * err_jv[self.cf],
                    self.sw : ksw * err_jv[self.sw]
                }                  
            elif self.state == 1:
                k = 0.5 * np.ones((6,1))
                return {
                    'lf' : k * err_jv['lf'],
                    'rf' : k * err_jv['rf']
                }
            elif self.state == 2: 
                # ksw = 60    * np.vstack(( 1, 1, 1, 1, 0, 0))
                # kcf = 41.67 * np.vstack(( 1, 1, 1, 1, 0, 0))
                ksw = 12    * np.vstack(( 1, 1, 1, 1, 0, 0))
                kcf = 12 * np.vstack(( 1, 1, 1, 1, 0, 0))
                return {
                    self.cf : kcf * err_jv[self.cf],
                    self.sw : ksw * err_jv[self.sw]
                }   

        def elasticTorq():
            '''state0 有自己的控制方法, 利用裝彈簧暫時hold住'''
            # ref_jp = np.vstack(( 0, 0, -0.37, 0.74, -0.37, 0))
            ref_jp = np.vstack((0,0,0,0,0,0))
            
            # kp = np.vstack(( 2, 2, 4, 6, 6, 4 )) # 裝彈簧
            # kp[4:] = np.vstack([0,0])
            # kp = np.vstack(( 0, 0, 0, 0, 0, 0 ))
            kp = np.vstack(( 1, 1, 2, 3, 3, 2 )) # 裝彈簧
            
            err_jp = {
                'lf' : ref_jp -self.jp[:6],
                'rf' : ref_jp -self.jp[6:]
            }

            return {
                'lf' : kp * err_jp['lf'],
                'rf' : kp * err_jp['rf']
            }
        
        def gravity( ) -> dict[str, np.ndarray]:
            '''計算重力矩, 則設加速度、速度=0'''
            jp_l, jp_r = self.jp[:6],self.jp[6:] #分別取出左右腳的由骨盆往下的joint_position
            jv_DS = np.zeros((6,1))
            ja_DS = np.zeros((6,1))
            jv_SS = np.zeros((12,1))
            ja_SS = np.zeros((12,1))
            
            #雙支撐的重力矩
            L_DS_gravity = np.reshape(pin.rnea(self.stance_l_model, self.stance_l_data, -jp_l[::-1], jv_DS, ja_DS ),(6,1))  
            R_DS_gravity = np.reshape(pin.rnea(self.stance_r_model, self.stance_r_data, -jp_r[::-1], jv_DS, ja_DS ),(6,1))  
            
            DS_gravity = np.vstack((- L_DS_gravity[::-1], - R_DS_gravity[::-1]))
            
            #右腳單支撐的重力矩
            jp_RSS = np.vstack((-jp_r[::-1],jp_l))
            Leg_RSS_gravity = np.reshape(pin.rnea(self.bipedal_r_model, self.bipedal_r_data, jp_RSS, jv_SS, ja_SS ),(12,1))  
            RSS_gravity = np.vstack(( Leg_RSS_gravity[6:], -Leg_RSS_gravity[5::-1] ))

            #左腳單支撐的重力矩
            jp_LSS = np.vstack(( -jp_l[::-1],jp_r ))

            Leg_LSS_gravity = np.reshape(pin.rnea(self.bipedal_l_model, self.bipedal_l_data, jp_LSS, jv_SS, ja_SS ),(12,1))  
            LSS_gravity = np.vstack(( -Leg_LSS_gravity[5::-1], Leg_LSS_gravity[6:] ))  

            #========================重力矩輸出==================================#
            if self.state == 30:
                Leg_gravity = 0.3*DS_gravity + 0.75*RSS_gravity if self.cf == 'rf' else\
                              0.3*DS_gravity + 0.75*LSS_gravity
            
            else:
                y_lf2pel = abs( pa_lf2pel_in_pf[1,0] )
                y_rf2pel = abs( pa_rf2pel_in_pf[1,0] )
                
                Leg_gravity = DS_gravity*y_lf2pel/0.1 + LSS_gravity*(0.1-y_lf2pel)/0.1 if y_lf2pel < y_rf2pel else\
                              DS_gravity*y_rf2pel/0.1 + LSS_gravity*(0.1-y_rf2pel)/0.1 if y_rf2pel < y_lf2pel else\
                              DS_gravity
            
            return {
                'lf':Leg_gravity[:6],
                'rf':Leg_gravity[6:]
            }
            # return {
            #     'lf':0.5 * LSS_gravity[:6] + 0.5 * RSS_gravity[:6],
            #     'rf':0.5 * LSS_gravity[6:] + 0.5 * RSS_gravity[6:]
            # }
            # return {
            #     'lf':LSS_gravity[:6],
            #     'rf':LSS_gravity[6:]
            # }
            
        def inertiaTorq(ja):
            '''目前慣性矩設成單位矩陣'''
            inertia_mat = np.identity(6) #慣量矩陣H
            return {
                'lf' : inertia_mat @ ja['lf'],
                'rf' : inertia_mat @ ja['rf']
            }
        
        #========================內環關節加速度==================================#
        ja = cal_jvErr_to_ja()
        
        #========================順向動力學==================================#
        gravity_torq = gravity()

        if self.state == 0: 
            elastic_torq = elasticTorq()
            return {
                'lf' : elastic_torq['lf'] + gravity_torq['lf'],
                'rf' : elastic_torq['rf'] + gravity_torq['rf'] 
            }

        elif self.state in [1, 2, 30]:
            inertia_torq = inertiaTorq(ja)
            return {
                # 'lf' : inertia_torq['lf'] + gravity_torq['lf'],
                'lf' : inertia_torq['lf'],
                # 'rf' : inertia_torq['rf'] + gravity_torq['rf']
                'rf' : inertia_torq['rf']
            }  
    
    def ankle_control(self):
        '''kd寫在xacro裡'''       
        ref_axy_ankle = np.zeros((2,1))
        axy_ft_in_wf = {
            'lf' : self.pt.a_lf_in_wf[:2],
            'rf' : self.pt.a_rf_in_wf[:2],
        }
        #===========================膝下兩關節進入加法器算誤差==================================#
        err_axy_ft_in_wf = {
            'lf': ref_axy_ankle - axy_ft_in_wf['lf'],
            'rf': ref_axy_ankle - axy_ft_in_wf['rf']
        }
        torq56 = np.vstack((
            PD_sw_y.send( err_axy_ft_in_wf[self.sw][1,0] ),
            PD_sw_x.send( err_axy_ft_in_wf[self.sw][0,0] )
        ))

        return torq56
    
    def estimatorcontrol(self):
        Kx = np.array([ [150, 15] ])
        # Kx = np.array([ [778, 23] ])
        # Kx = np.array([ [ 290.3274, 15.0198 ] ])*0.5
        # Ky = np.array([[ -177.0596, 9.6014 ]])*0.15
        # Ky = np.array([[ -1664,37.3 ]])
        Ky = np.array([[ -150, 15 ]])
        
        ux = {
            'lf' : -Kx @ (self.pt.xLy_com_in_lf - self.ref_xLy_com_in_lf),
            'rf' : -Kx @ (self.pt.xLy_com_in_rf - self.ref_xLy_com_in_rf)
        }
        uy = {
            'lf' : -Ky @ (self.pt.yLx_com_in_lf - self.ref_yLx_com_in_lf),
            'rf' : -Ky @ (self.pt.yLx_com_in_rf - self.ref_yLx_com_in_rf)
        }

        torq56 = {
            'lf' : - np.vstack(( uy['lf'], ux['lf'] )),
            'rf' : - np.vstack(( uy['rf'], ux['rf'] ))
        }
        
        for ft in ['lf', 'rf']:
            for i in [0, 1]:
                torq56[ft][i,0] = min(20, torq56[ft][i,0])
                torq56[ft][i,0] = max(-20, torq56[ft][i,0])

        return torq56
        
        '''kd寫在xacro裡'''       
        ref_axy_ankle = np.zeros((2,1))
        axy_ft_in_wf = {
            'lf' : self.pt.a_lf_in_wf[:2],
            'rf' : self.pt.a_rf_in_wf[:2],
        }
        #===========================膝下兩關節進入加法器算誤差==================================#
        err_axy_ft_in_wf = {
            'lf': ref_axy_ankle - axy_ft_in_wf['lf'],
            'rf': ref_axy_ankle - axy_ft_in_wf['rf']
        }
        torq56 = np.vstack((
            PD_cf_y.send( err_axy_ft_in_wf[self.cf][1,0] ),
            PD_cf_x.send( err_axy_ft_in_wf[self.cf][0,0] )
        ))

        return torq56
    
    def estimatorcontrol_oldversion(self):            
        #===========================離散版狀態矩陣==================================#
        #x, Ly
        Ax = np.array([
            [1,      0.00247],
            [0.8832, 1]
            ])
        Bx = np.vstack(( 0, 0.01 ))
                
        Kx = np.array([[ 290.3274, 15.0198 ]])*0.5
        Lx = np.array([
            [0.1390, 0.0025],
            [0.8832, 0.2803]
            ])
        #------------------------------------------------------------------------#
        #y, Lx
        Ay = np.array([
            [1,      -0.00247],
            [-0.8832, 1]
            ])
        By = np.vstack(( 0,0.01 ))
        
        Ky = np.array([[ -177.0596, 9.6014 ]])*0.15
        Ly = np.array([
            [ 0.1288, -0.0026],
            [-0.8832,  0.1480]
            ])
        
        #===========================離散版補償器：全都用過去的值==================================#
        oberr_xLy_com_in_ft_past = {
            'lf' : self.pt.xLy_com_in_lf_past - self.ob_xLy_com_in_lf_past, #計算估測誤差
            'rf' : self.pt.xLy_com_in_rf_past - self.ob_xLy_com_in_rf_past 
        }
        oberr_yLx_com_in_ft_past = {
            'lf' : self.pt.yLx_com_in_lf_past - self.ob_yLx_com_in_lf_past, #計算估測誤差
            'rf' : self.pt.yLx_com_in_rf_past - self.ob_yLx_com_in_rf_past 
        }
        
        
        # xe1=Ax*e0+B*u0+L*err0 (1是現在,0是過去)
        ob_xLy_com_in_ft = {
            'lf' : Ax @ self.ob_xLy_com_in_lf_past + Bx @ self.ux_lf_past + Lx @ oberr_xLy_com_in_ft_past['lf'],
            'rf' : Ax @ self.ob_xLy_com_in_rf_past + Bx @ self.ux_rf_past + Lx @ oberr_xLy_com_in_ft_past['rf']
        }
        ob_yLx_com_in_ft = {
            'lf' : Ay @ self.ob_yLx_com_in_lf_past + By @ self.ux_lf_past + Ly @ oberr_yLx_com_in_ft_past['lf'],
            'rf' : Ay @ self.ob_yLx_com_in_rf_past + By @ self.ux_rf_past + Ly @ oberr_yLx_com_in_ft_past['rf']
        }
        
        #===========================現在的輸入==================================#
        if self.state == 30 and self.cf != self.cf_past: 
            self.ref_xLy_com_in_lf = ob_xLy_com_in_ft['lf']
            self.ref_yLx_com_in_lf = ob_xLy_com_in_ft['rf']
            self.ref_xLy_com_in_rf = ob_yLx_com_in_ft['lf']
            self.ref_yLx_com_in_rf = ob_yLx_com_in_ft['rf']
            
        ux = {
            'lf' : -Kx @ (ob_xLy_com_in_ft['lf'] - self.ref_xLy_com_in_lf),
            'rf' : -Kx @ (ob_xLy_com_in_ft['rf'] - self.ref_xLy_com_in_rf)
        }
        uy = {
            'lf' : -Ky @ (ob_yLx_com_in_ft['lf'] - self.ref_yLx_com_in_lf),
            'rf' : -Ky @ (ob_yLx_com_in_ft['rf'] - self.ref_yLx_com_in_rf)
        }

        
        
        #===========================若是在切換瞬間, 則強迫支撐腳角動量連續,把扭矩切成0來避免腳沒踩穩==================================#

        if self.state == 30 and self.cf != self.cf_past: 
            self.pt.__dict__[f"xLy_com_in_{self.cf}"][1,0] = deepcopy(self.pt.__dict__[f"xLy_com_in_{self.sw}_past"][1,0]) #擺動腳的過去是支撐腳
            self.pt.__dict__[f"yLx_com_in_{self.cf}"][1,0] = deepcopy(self.pt.__dict__[f"yLx_com_in_{self.sw}_past"][1,0]) #擺動腳的過去是支撐腳
            
            ob_xLy_com_in_ft[self.cf][1,0] = self.__dict__[f"ob_xLy_com_in_{self.sw}_past"][1,0]
            ob_yLx_com_in_ft[self.cf][1,0] = self.__dict__[f"ob_yLx_com_in_{self.sw}_past"][1,0]
            
            ux[self.cf] = np.zeros((1,1))
            uy[self.cf] = np.zeros((1,1))

        #===========================更新舊值==================================#
        self.ux_lf_past = ux['lf']
        self.ux_rf_past = ux['rf']
        self.uy_lf_past = uy['lf']
        self.uy_rf_past = uy['rf']
        
        self.pt.xLy_com_in_lf_past = self.pt.xLy_com_in_lf
        self.pt.xLy_com_in_rf_past = self.pt.xLy_com_in_rf
        self.pt.yLx_com_in_lf_past = self.pt.yLx_com_in_lf
        self.pt.yLx_com_in_rf_past = self.pt.yLx_com_in_rf
        
        self.ob_xLy_com_in_lf_past = ob_xLy_com_in_ft['lf']
        self.ob_xLy_com_in_rf_past = ob_xLy_com_in_ft['rf']
        self.ob_yLx_com_in_lf_past = ob_yLx_com_in_ft['lf']
        self.ob_yLx_com_in_rf_past = ob_yLx_com_in_ft['rf']
        
        #===========================現在的力矩==================================#
        torq56 = {
            'lf' : - np.vstack(( uy['lf'], ux['lf'] )),
            'rf' : - np.vstack(( uy['rf'], ux['rf'] ))
        }
        
        torq56['lf'][0,0] = 20 if torq56['lf'][0,0]> 20 else\
                           -20 if torq56['lf'][0,0]< -20 else\
                            torq56['lf'][0,0]
        torq56['lf'][1,0] = 20 if torq56['lf'][1,0]> 20 else\
                           -20 if torq56['lf'][1,0]< -20 else\
                            torq56['lf'][1,0]
        torq56['rf'][0,0] = 20 if torq56['rf'][0,0]> 20 else\
                           -20 if torq56['rf'][0,0]< -20 else\
                            torq56['rf'][0,0]
        torq56['rf'][1,0] = 20 if torq56['rf'][1,0]> 20 else\
                           -20 if torq56['rf'][1,0]< -20 else\
                            torq56['rf'][1,0]                    
            
        return torq56

dirname = '/home/ldsc/Desktop/'    
def createfile():
    traj_name = ['traj0.txt', 'traj1.txt', 'traj30.txt']
    torq_names = ['torq0.txt', 'torq1.txt', 'torq2.txt', 'torq30.txt']
    meaL_names = ['meaL0.txt', 'meaL1.txt', 'meaL2.txt', 'meaL30.txt']
    jp_names = ['jp0.txt', 'jp2.txt']
    jv_names = ['jv0.txt', 'jv2.txt']
    for name in torq_names + meaL_names + jp_names + jv_names:
        with open( dirname + name ,'w') as fh:
            pass
    
def writetorq(self,torq):
    torq_name = f'torq{int(self.state)}.txt'
    L = np.vstack((torq['lf'], torq['rf'])).flatten()
    with open( dirname + torq_name ,'a') as fh:
        fh.write("{} {} {} {} {} {} {} {} {} {} {} {}\n".format(*L))
        
def write_measure_L(self):
    meaLname = f'meaL{int(self.state)}.txt'
    L = np.vstack(( self.pt.xLy_com_in_lf,self.pt.yLx_com_in_lf )).flatten()
    with open( dirname + meaLname ,'a') as fh:
        fh.write("{} {} {} {}\n".format(*L))
        
def write_jp(self):
    meaLname = f'jp{int(self.state)}.txt'
    L = self.jp.flatten()
    with open( dirname + meaLname ,'a') as fh:
        fh.write("{} {} {} {} {} {} {} {} {} {} {} {}\n".format(*L))
        
def write_jv(self,cmd_jv):
    meaLname = f'jv{int(self.state)}.txt'
    L = self.jv.flatten()
    L1 = np.vstack((cmd_jv['lf'],cmd_jv['rf'])).flatten()
    with open( dirname + meaLname ,'a') as fh:
        fh.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(*L,*L1))

if __name__ == '__main__':
    createfile()
    rclpy.init(args = None)

    upper_level_controllers = UpperLevelController()
    rclpy.spin(upper_level_controllers)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    upper_level_controllers.destroy_node()
    rclpy.shutdown()