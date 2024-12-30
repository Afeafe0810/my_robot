import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray 

import pinocchio as pin

import numpy as np; np.set_printoptions(precision=2)

import copy
import math
from scipy.spatial.transform import Rotation as R

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
        #建立ROS的node
        super().__init__('upper_level_controllers')
        
        #負責模型與ROS的功能
        self.ros = ROSInterfaces(self, self.main_controller_callback)
        
        #負責量測各部位的位置與姿態
        self.frame = RobotFrame()
        
        #負責處理軌跡
        self.traj = Trajatory()
        
        #==============================================================robot constant==============================================================#     
        self.stance = 2
        self.stance_past = 2
        self.DS_time = 0.0

        #==============================================================robot frame==============================================================#     

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


 
        # Initialize the service client
        self.attach_link_client = self.create_client(AttachLink, '/ATTACHLINK')
        self.detach_link_client = self.create_client(DetachLink, '/DETACHLINK') 

    def stance_change(self, state, stance, contact_t):
        stance = 1 if state == 0 else\
                 1 if state == 1 else\
                 1 if state == 2 else\
                stance #先不變

        if state == 2:
            if self.DS_time <= 10 * Config.DDT:
                self.DS_time += Config.TIMER_PERIOD
                print("DS",self.DS_time)

        # 時間到就做兩隻腳的切換
        if state == 30 and abs(contact_t-0.5) <= 0.005 :
            stance = 0 if stance == 1 else\
                     1 #if self.stance == 0
                     
        return stance  

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
 
    def main_controller_callback(self):
        #==========拿取訂閱值==========#
        p_base_in_wf, r_base_to_wf, state, contact_lf, contact_rf, jp, jv = self.ros.updateSubData()
        
        #==========更新可視化的機器人==========#
        config = self.ros.update_VizAndMesh(jp)
        
        #==========更新frame==========#
        self.frame.updateFrame(self.ros, config, p_base_in_wf, r_base_to_wf, jp)

        px_in_lf,px_in_rf = self.frame.get_posture(self.frame.pa_pel_in_pf, self.frame.pa_lf_in_pf, self.frame.pa_rf_in_pf)
        
        #==========待刪掉==========#
        self.P_B_wf, self.O_wfB, self.pub_state, self.l_contact, self.r_contact, self.jp_sub = p_base_in_wf, r_base_to_wf, state, contact_lf, contact_rf, jp
        l_contact,r_contact = self.l_contact, self.r_contact
        
        #========接觸判斷========#
        l_contact = (self.frame.p_lf_in_wf[2,0] <= 0.01)
        r_contact = (self.frame.p_rf_in_wf[2,0] <= 0.01)

        #========支撐狀態切換=====#
        stance = self.stance_change(state, state, self.contact_t)
        
        #========軌跡規劃========#
        ref_pa_pel_in_wf, ref_pa_lf_in_wf, ref_pa_rf_in_wf = trajRef_planning(state, self.DS_time, Config.DDT)
        
        #========重力補償跟kl,kr========#
        l_leg_gravity, r_leg_gravity, kl, kr = gravity_compemsate(self.ros, jp, stance, px_in_lf, px_in_rf, l_contact, r_contact, state)
        
        self.ros.publisher["gravity_l"].publish(Float64MultiArray(data=l_leg_gravity))
        self.ros.publisher["gravity_r"].publish(Float64MultiArray(data=r_leg_gravity))
        
        #========膝上雙環控制========#
        #--------膝上外環控制--------#
        JLL, JRR =  self.frame.left_leg_jacobian()
        
        Le_2,Re_2 = endErr_to_endVel(self.frame, ref_pa_pel_in_wf, ref_pa_lf_in_wf, ref_pa_rf_in_wf)
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
            
            torque[sf][4:6] = swingAnkle_PDcontrol(stance, self.frame.r_lf_to_wf, self.frame.r_rf_to_wf)
            torque[cf][4:6] = alip_control(self.frame, stance, self.stance_past, self.frame.p_com_in_wf, self.frame.p_lf_in_wf, self.frame.p_rf_in_wf, ref_pa_pel_in_wf, ref_pa_lf_in_wf,ref_pa_rf_in_wf)
            if stance == 1:
                self.ros.publisher["torque_l"].publish( Float64MultiArray(data = torque['lf'][4:6] ))
            # torque_R =  alip_R(self, stance,px_in_lf,torque_ALIP,com_in_rf,state)
            self.ros.publisher['effort'].publish(Float64MultiArray(data = np.vstack(( torque['lf'], torque['rf'] )) ) )
            
        elif state == 2:
            torque = innerloopDynamics(jv,VL,VR,l_leg_gravity,r_leg_gravity,kl,kr)
            
            torque[sf][4:6] = swingAnkle_PDcontrol(stance, self.frame.r_lf_to_wf, self.frame.r_rf_to_wf)
            torque[cf][4:6] = alip_control(self.frame, stance, self.stance_past, self.frame.p_com_in_wf, self.frame.p_lf_in_wf, self.frame.p_rf_in_wf, ref_pa_pel_in_wf, ref_pa_lf_in_wf,ref_pa_rf_in_wf)
            if stance == 1:
                self.ros.publisher["torque_l"].publish( Float64MultiArray(data = torque['lf'][4:6] ))
            # torque_R =  alip_R(self, stance,px_in_lf,torque_ALIP,com_in_rf,state)
            self.ros.publisher['effort'].publish(Float64MultiArray(data = np.vstack(( torque['lf'], torque['rf'] )) ) )

        # elif state == 30:
        #     # self.to_matlab()
        #     torque_ALIP = walking_by_ALIP(self, jv, VL, VR, l_leg_gravity, r_leg_gravity, kl, kr, self.O_wfL, self.O_wfR)
        #     torque_L =  alip_L(self, stance, torque_ALIP, ref_pa_pel_in_wf, ref_pa_lf_in_wf)
        #     # torque_R =  alip_R(self, stance,px_in_lf,torque_ALIP,com_in_rf,state)
        #     # print(stance)
            
        #     if stance == 1:
        #         self.ros.publisher['effort'].publish(Float64MultiArray(data=torque_L))

        #     elif stance == 0:
        #         self.ros.publisher['effort'].publish(Float64MultiArray(data=torque_R))
        #     # self.ros.publisher['effort'].publish(Float64MultiArray(data=torque_ALIP))
        
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
