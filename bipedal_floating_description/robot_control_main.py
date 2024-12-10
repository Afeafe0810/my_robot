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

#================ import other code =====================#
from utils.robot_control_init import ULC_init
from utils.robot_control_sensor import ULC_sensor
from utils.robot_control_traj import ULC_traj
from utils.robot_control_knee_control import Outterloop, Innerloop
#========================================================#
        


class UpperLevelController(Node):

    def __init__(self):
        
        super().__init__('upper_level_controllers')
        ULC_init.create_publishers(self)
        ULC_init.create_subscribers(self)
        self.robot = ULC_init.load_URDF(self,"/home/ldsc/ros2_ws/src/bipedal_floating_description/urdf/bipedal_floating.pin.urdf")
        self.call = 0

        #joint_state(subscribe data)
        self.jp_sub = np.zeros(12)
        self.jv_sub = np.zeros(12)

        #joint_velocity_cal
        self.joint_position_past = np.zeros((12,1))

        #joint_velocity_filter (jp = after filter)
        self.jp = np.zeros((12,1))
        self.jp_p = np.zeros((12,1))
        self.jp_pp = np.zeros((12,1))
        self.jp_sub_p = np.zeros((12,1))
        self.jp_sub_pp = np.zeros((12,1))

        #joint_velocity_filter (jv = after filter)
        self.jv = np.zeros((12,1))
        self.jv_p = np.zeros((12,1))
        self.jv_pp = np.zeros((12,1))
        self.jv_sub_p = np.zeros((12,1))
        self.jv_sub_pp = np.zeros((12,1))

        #position PI
        self.Le_dot_past = 0.0
        self.Le_past = 0.0
        self.Re_dot_past = 0.0
        self.Re_past = 0.0
        
        #state by user
        self.pub_state = 0
        #contact by bumper
        self.l_contact = 1
        self.r_contact = 1



        
        # Initialize meschcat visualizer
        self.viz = pin.visualize.MeshcatVisualizer(
            self.robot.model, self.robot.collision_model, self.robot.visual_model
        )
        self.robot.setVisualizer(self.viz, init=False)
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel()

        #


        # Set initial robot configuration
        print(self.robot.model)
        print(self.robot.q0)
        self.init_configuration = pink.Configuration(self.robot.model, self.robot.data, self.robot.q0)
        self.viz.display(self.init_configuration.q)

        self.timer_period = 0.01 # seconds 跟joint state update rate&取樣次數有關
        # self.timer = self.create_timer(self.timer_period, self.main_controller_callback)

        self.tt = 0
        self.P_Y_ref = 0.0

        self.count = 0
        self.stance = 2
        self.stance_past = 2
        self.DS_time = 0.0
        self.RSS_time = 0.0
        self.LSS_time = 0.0
        self.RSS_count = 0
        self.DDT = 2
        self.RDT = 1
        self.LDT = 1

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

        self.delay = 0

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
        self.Vy_L = 0.0
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
        #--compensator
        self.ob_x_L = np.zeros((2,1))
        self.ob_x_past_L = np.zeros((2,1))
        self.ob_y_L = np.zeros((2,1))
        self.ob_y_past_L = np.zeros((2,1))
        self.ob_x_R = np.zeros((2,1))
        self.ob_x_past_R = np.zeros((2,1))
        self.ob_y_R = np.zeros((2,1))
        self.ob_y_past_R = np.zeros((2,1))
        #--torque
        self.ap_L = 0.0
        self.ap_past_L = 0.0
        self.ar_L = 0.0
        self.ar_past_L = 0.0
        self.ap_R = 0.0
        self.ap_past_R = 0.0
        self.ar_R = 0.0
        self.ar_past_R = 0.0

        #--ref    
        self.ref_x_L = np.zeros((2,1))
        self.ref_y_L = np.zeros((2,1))
        self.ref_x_R = np.zeros((2,1))
        self.ref_y_R = np.zeros((2,1))

        #touch for am tracking check
        self.touch = 0
 
        # Initialize the service client


    def contact_collect(self):
        '''
            只複製並回傳(l_contact,r_contact)
        '''
        l_contact = copy.deepcopy(self.l_contact)
        r_contact = copy.deepcopy(self.r_contact)
        # print("L:",l_contact,"R:",r_contact)

        return l_contact,r_contact

    def state_collect(self):
        self.state_current = copy.deepcopy(self.pub_state)

        return self.state_current
    

        
    def collect_joint_data(self):
        '''
        就只是收集而已
        '''
        joint_position = copy.deepcopy(self.jp_sub)
        joint_velocity = copy.deepcopy(self.jv_sub)

        joint_position = np.reshape(joint_position,(12,1))
        joint_velocity = np.reshape(joint_velocity,(12,1))

        return joint_position,joint_velocity

    





   

         

 


    def stance_change(self,state,px_in_lf,px_in_rf,stance,contact_t):
        '''
        state0利用雙腳左右距離來判斷是哪隻腳支撐/雙支撐(骨盆距某腳0.06內為支撐腳，其他狀態是雙支撐)
        state1...剩下的都看不太懂
        '''
        if state == 0:
            #用骨盆相對左右腳掌位置來切換   
            if abs(px_in_lf[1,0])<=0.06:
                stance = 1 #左單支撐
            elif abs(px_in_rf[1,0])<=0.06:
                stance = 1 #右單支撐
            else:
                stance = 2 #雙支撐

        if state == 1:
            if self.DS_time <= self.DDT:## \\\\DS_time和DDT是什麼？？？
                self.DS_time += self.timer_period
                stance = 2
                print("DS",self.DS_time)
            else:
                self.DS_time = 10.1
                stance = 1
                self.RSS_time = 0.01

        if state == 2:
            if stance == 2:
                if self.DS_time <= self.DDT:
                    stance = 2
                    self.DS_time += self.timer_period
                else:
                    self.DS_time = 0.0
                    if abs(px_in_lf[1,0])<=0.08:
                        stance = 1 #左單支撐
                        self.LSS_time = 0.01
                    elif abs(px_in_rf[1,0])<=0.08:
                        stance = 0 #右單支撐
                        self.RSS_time = 0.01
            if stance == 0:
                if self.RSS_time <= self.RDT:
                    stance = 0
                    self.RSS_time += self.timer_period
                else:
                    stance = 2 #雙支撐
                    self.DS_time = 0.01
                    self.RSS_time = 0
                    self.RSS_count = 1
            if stance == 1:
                if self.LSS_time <= self.LDT:
                    stance = 1
                    self.LSS_time += self.timer_period
                else:
                    stance = 2 #雙支撐
                    self.DS_time = 0.01
                    self.LSS_time = 0
                    self.RSS_count = 0
        
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








    
   

    
   
  

    def main_controller_callback(self):
        
        joint_position,joint_velocity = self.collect_joint_data()
        joint_velocity_cal = ULC_sensor.joint_velocity_cal(joint_position)
        jv_f = ULC_sensor.joint_velocity_filter(joint_velocity_cal)

        configuration = pink.Configuration(self.robot.model, self.robot.data,joint_position)
        self.viz.display(configuration.q)

        #從pink拿相對base_frame的位置及姿態角  ////我覺得是相對pf吧
        ULC_sensor.get_position_pf(self,configuration)
        px_in_lf,px_in_rf = ULC_sensor.get_posture(self)
        com_in_lf,com_in_rf,com_in_pink = ULC_sensor.com_position(self,joint_position)
        #算wf下的位置及姿態
        ULC_sensor.pelvis_in_wf(self)
        ULC_sensor.data_in_wf(self, com_in_pink)
        #這邊算相對的矩陣
        ULC_sensor.rotation_matrix(self, joint_position)
        #這邊算wf下各軸姿態
        ULC_sensor.relative_axis(self)

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

        #怎麼切支撐狀態要改!!!!!
        stance = self.stance_change(state,px_in_lf,px_in_rf,self.stance,self.contact_t)
        if state == 30:
            stance = 0
        else:
            ULC_traj.ref_cmd(self, state,px_in_lf,px_in_rf,stance,com_in_lf,com_in_rf)
            l_leg_gravity,r_leg_gravity,kl,kr = Innerloop.gravity_compemsate(self, joint_position,stance,px_in_lf,px_in_rf,l_contact,r_contact,state)
        
        
        

        

        JLL = Outterloop.left_leg_jacobian(self)
        JRR = Outterloop.right_leg_jacobian(self)
        Le_2,Re_2 = Outterloop.calculate_err(self,state)
        VL,VR = Innerloop.velocity_cmd(self,Le_2,Re_2,jv_f,stance,state)
        
        #control
        if state == 0:   
            Innerloop.balance(self, joint_position,l_leg_gravity,r_leg_gravity)

        elif state == 1 or state == 2:
            torque_kine = Innerloop.swing_leg(self, jv_f,VL,VR,l_leg_gravity,r_leg_gravity,kl,kr)
            # self.effort_publisher.publish(Float64MultiArray(data=torque_kine))
            
            #更新量測值
            torque_ALIP = Innerloop.walking_by_ALIP(self,jv_f,VL,VR,l_leg_gravity,r_leg_gravity,kl,kr)
            torque_L =  Innerloop.alip_L(self,stance,px_in_lf,torque_ALIP,com_in_lf,state)
            torque_R =  Innerloop.alip_R(self,stance,px_in_lf,torque_ALIP,com_in_rf,state)
            self.effort_publisher.publish(Float64MultiArray(data=torque_L))

        elif state == 30:
            # self.to_matlab()
            torque_ALIP = Innerloop.walking_by_ALIP(self,jv_f,VL,VR,l_leg_gravity,r_leg_gravity,kl,kr)
            torque_L =  Innerloop.alip_L(self,stance,px_in_lf,torque_ALIP,com_in_lf,state)
            torque_R =  Innerloop.alip_R(self,stance,px_in_lf,torque_ALIP,com_in_rf,state)
            # print(stance)
            if stance == 1:
                self.effort_publisher.publish(Float64MultiArray(data=torque_L))

            elif stance == 0:
                self.effort_publisher.publish(Float64MultiArray(data=torque_R))
            # self.effort_publisher.publish(Float64MultiArray(data=torque_ALIP))
        
        self.state_past = copy.deepcopy(state)
        self.stance_past = copy.deepcopy(stance)
        
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