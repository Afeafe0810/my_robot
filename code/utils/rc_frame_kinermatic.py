from std_msgs.msg import Float64MultiArray 
import numpy as np; np.set_printoptions(precision=2)
import copy

from utils.config import Config

class RobotFrame:
    def __init__(self):
        #==========微分器==========#
        self.diffter_p_com_in_wf = Diffter()
        #==========濾波器==========#
        self.filter_v_com_in_wf = Filter()
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
        pass
   
    # def alip_R(self,stance_type,px_in_rf,torque_ALIP,com_in_rf,state):
    #     # print("ALIP_R")

    #     torque = copy.deepcopy(torque_ALIP) 
    #     stance = copy.deepcopy(stance_type) 
    #     #獲取量測值(相對於右腳腳底)
    #     # PX_r = copy.deepcopy(com_in_rf)
    #     com_in_wf = copy.deepcopy(self.P_COM_wf)
    #     rx_in_wf = copy.deepcopy(self.P_R_wf)
    #     PX_r = com_in_wf - rx_in_wf
        
    #     #計算質心速度
    #     self.CX_dot_R = (com_in_wf[0,0] - self.CX_past_R)/self.timer_period
    #     self.CX_past_R = com_in_wf[0,0]
    #     self.CY_dot_R = (com_in_wf[1,0] - self.CY_past_R)/self.timer_period
    #     self.CY_past_R = com_in_wf[1,0]

    #     #velocity filter
    #     self.Vx_R = 0.7408*self.Vx_past_R + 0.2592*self.CX_dot_past_R  #濾過後的速度(5Hz)
    #     self.Vx_past_R = self.Vx_R
    #     self.CX_dot_past_R =  self.CX_dot_R

    #     self.Vy_R = 0.7408*self.Vy_past_R + 0.2592*self.CY_dot_past_R  #濾過後的速度(5Hz)
    #     self.Vy_past_R = self.Vy_R
    #     self.CY_dot_past_R =  self.CY_dot_R

    #     #量測值
    #     Xc_mea = PX_r[0,0]
    #     Ly_mea = 9*self.Vx_R*0.45
    #     Yc_mea = PX_r[1,0]
    #     Lx_mea = -9*self.Vy_R*0.45 #(記得加負號)
    #     self.mea_x_R = np.array([[Xc_mea],[Ly_mea]])
    #     self.mea_y_R = np.array([[Yc_mea],[Lx_mea]])

    #     #參考值(直接拿從online_planning來的)
    #     ref_x_R = 0
    #     ref_y_R = 0.1
    #     # self.PX_ref = np.array([[0.0],[0.0],[0.57],[0.0],[0.0],[0.0]])
    #     # self.LX_ref = np.array([[0.0],[0.1],[0.0],[0.0],[0.0],[0.0]])
    #     # self.RX_ref = np.array([[0.0],[-0.1],[0.0],[0.0],[0.0],[0.0]])

    #     #xc & ly model(m=9 H=0.45 Ts=0.01)
    #     Ax = np.array([[1,0.00247],[0.8832,1]])
    #     Bx = np.array([[0],[0.01]])
    #     Cx = np.array([[1,0],[0,1]])  
    #     #--LQR
    #     Kx = np.array([[290.3274,15.0198]])
    #     Lx = np.array([[0.1390,0.0025],[0.8832,0.2803]]) 
    #     # Kx = np.array([[184.7274,9.9032]])
    #     # Lx = np.array([[0.1427,-0.0131],[0.8989,0.1427]]) 
        
    #     #--compensator
    #     self.ob_x_R = Ax@self.ob_x_past_R + self.ap_past_R*Bx + Lx@(self.mea_x_past_R - Cx@self.ob_x_past_R)

    #     #由於程式邏輯 使得右腳在擺動過程也會估測 然而並不會拿來使用
    #     #為了確保支撐腳切換過程 角動量估測連續性
    #     if self.stance_past == 1 and self.stance == 0:
    #         self.mea_x_R[1,0] = copy.deepcopy(self.mea_x_past_L[1,0])
    #         self.ob_x_R[1,0] = copy.deepcopy(self.ob_x_past_L[1,0])
        
    #     #----calculate toruqe
    #     # self.ap_R = -Kx@(self.ob_x_R)  #(地面給機器人 所以使用時要加負號)
    #     # self.ap_R = -torque[10,0] #torque[10,0]為右腳pitch對地,所以要加負號才會變成地對機器人
    #     self.ap_R = -Kx@(self.ob_x_R - ref_x_R)*0.5

    #     # if self.ap_R >= 3:
    #     #     self.ap_R =3
    #     # elif self.ap_R <= -3:
    #     #     self.ap_R =-3

    #     #切換瞬間 扭矩切成0 避免腳沒踩穩
    #     if self.stance_past == 1 and self.stance == 0:
    #         self.ap_R = 0
        
    #     #--torque assign
    #     torque[10,0] = -self.ap_R
    #     #----update
    #     self.mea_x_past_R = self.mea_x_R
    #     self.ob_x_past_R = self.ob_x_R
    #     self.ap_past_R = self.ap_R

    #     #yc & lx model
    #     Ay = np.array([[1,-0.00247],[-0.8832,1]])
    #     By = np.array([[0],[0.01]])
    #     Cy = np.array([[1,0],[0,1]])  
    #     #--LQR
    #     # Ky = np.array([[-290.3274,15.0198]])
    #     # Ly = np.array([[0.1390,-0.0025],[-0.8832,0.2803]])
    #     Ky = np.array([[-177.0596,9.6014]])
    #     Ly = np.array([[0.1288,-0.0026],[-0.8832,0.1480]])
    #     #--compensator
    #     self.ob_y_R = Ay@self.ob_y_past_R + self.ar_past_R*By + Ly@(self.mea_y_past_R - Cy@self.ob_y_past_R)

    #     #由於程式邏輯 使得右腳在擺動過程也會估測 然而並不會拿來使用
    #     #為了確保支撐腳切換過程 角動量估測連續性
    #     if self.stance_past == 1 and self.stance == 0:
    #         self.mea_y_R[1,0] = copy.deepcopy(self.mea_y_past_L[1,0])
    #         self.ob_y_R[1,0] = copy.deepcopy(self.ob_y_past_L[1,0])

    #     #----calculate toruqe
    #     # self.ar_R = -Ky@(self.ob_y_R)
    #     # self.ar_R = -torque[11,0]#torque[11,0]為右腳roll對地,所以要加負號才會變成地對機器人
    #     self.ar_R = -Ky@(self.ob_y_R - ref_y_R)*0.15

    #     #切換瞬間 扭矩切成0 避免腳沒踩穩
    #     if self.stance_past == 1 and self.stance == 0:
    #         self.ar_R = 0

    #     # if self.ar_R >= 3:
    #     #     self.ar_R =3
    #     # elif self.ar_R <= -3:
    #     #     self.ar_R =-3

    #     #--torque assign
    #     torque[11,0] = -self.ar_R
    #     #----update
    #     self.mea_y_past_R = self.mea_y_R
    #     self.ob_y_past_R = self.ob_y_R
    #     self.ar_past_R = self.ar_R

    #     return torque


class Filter:
    def __init__(self):
        self.__isStarted = True
        self.__u_p = None
        self.__y_p = None
        # self.__u_pp = None
        # self.__y_pp = None
        
    def filt(self, u):
        if self.__isStarted:
            self.__isStarted = False
            self.__u_p = 0*u
            self.__y_p = 0*u
        
        y = 0.7408 * self.__y_p + 0.2592 * self.__u_p
        #update
        self.__y_p = y
        self.__u_p = u
        
        return  y

class Diffter:
    def __init__(self):
        self.__isStarted = True
        self.__u_p = None
        
    def diff(self, u):
        if self.__isStarted:
            self.__isStarted = False
            self.__u_p = 0*u
            
        y = ( u - self.__u_p ) / Config.TIMER_PERIOD
        self.__u_p = u
        return  y