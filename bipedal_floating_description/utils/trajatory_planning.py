import numpy as np; np.set_printoptions(precision=2)
from math import cosh, sinh, cos, sin
#================ import library ========================#
from utils.config import Config
#========================================================#


class Trajatory:
    def __init__(self):
        self.ref_pa_in_wf = {
            'lf' : np.zeros((6,1)),
            'rf' : np.zeros((6,1)),
            'pel': np.zeros((6,1)),
        }
        pass

def trajRef_planning(state,DS_time, DDT):
    if state == 0: #假雙支撐
        return __bipedalBalanceTraj()
    
    elif state == 1: #真雙支撐
        return __bipedalBalanceTraj()
    
    elif state == 2: #骨盆移到支撐腳
        return __comMoveTolf(DS_time, DDT)
    
    elif state == 30: #ALIP規劃
        
        pass
    
def __bipedalBalanceTraj():
    ref_p_pel_in_wf = np.array([[0.0],[0.0],[0.57],[0.0],[0.0],[0.0]])
    ref_p_lf_in_wf = np.array([[0.0],[0.1],[0.0],[0.0],[0.0],[0.0]])
    ref_p_rf_in_wf = np.array([[0.0],[-0.1],[0.0],[0.0],[0.0],[0.0]])
    
    return {
        'pel': ref_p_pel_in_wf, 
        'lf' : ref_p_lf_in_wf ,
        'rf' : ref_p_rf_in_wf ,
    }
    
def __comMoveTolf(DS_time, DDT):
    ref_pa_lf_in_wf  =  np.vstack(( 0,  0.1,  0,   0, 0, 0 ))
    ref_pa_rf_in_wf  =  np.vstack(( 0, -0.1,  0,   0, 0, 0 )) if 0 < DS_time <= DDT else \
                        np.vstack(( 0, -0.1,  0.05* (DS_time-0*DDT)/(0.2*DDT), 0, 0, 0 )) if DDT < DS_time <= 0.2*DDT else\
                        np.vstack(( 0, -0.1,  0.05, 0, 0, 0 ))
                        # np.vstack(( 0, -0.1,  0.05 - 0.05* (DS_time-0.8*DDT)/(0.2*DDT), 0, 0, 0 )) if 0.8*DDT < DS_time <= DDT else\
                        # np.vstack(( 0, -0.1,  0, 0, 0, 0 ))
    
    # ref_pa_pel_in_wf = np.vstack(( 0, 0.06*DS_time/( 0.5*DDT ), 0.55, 0, 0, 0 )) if 0 < DS_time <= 0.5*DDT else \
    #                   np.vstack(( 0, 0.06,                     0.55, 0, 0, 0 ))
    
    ref_pa_pel_in_wf = np.vstack(( 0, 0.09*DS_time/( 0.5*DDT ), 0.55, 0, 0, 0 )) if 0 < DS_time <= 0.5*DDT else \
                      np.vstack(( 0, 0.09,                     0.55, 0, 0, 0 ))
                      
    return {
        'pel': ref_pa_pel_in_wf, 
        'lf' : ref_pa_lf_in_wf ,
        'rf' : ref_pa_rf_in_wf ,
    }

class ALIP_traj:
    def plan(cls, stance, des_vx_com_in_cf_2T, t, X0, Y0):
        cf, sf = stance
        
        m = Config.MASS
        H = Config.IDEAL_Z_COM_IN_WF
        w = Config.OMEGA
        T = Config.STEP_TIMELENGTH
        W = Config.IDEAL_Y_RFTOLF_IN_WF
        h = Config.STEP_HEIGHT
        sign = -1 if sf == 'rf' else\
                1
        
        #下兩步的理想角動量
        des_Ly_com_in_cf_2T = m * des_vx_com_in_cf_2T * H
        des_Lx_com_in_cf_2T = sign * ( 0.5*m*H*W ) * ( w*sinh(w*T) ) / ( 1+cosh(w*T) ) 
        
        #現在的質心ref
        X0, Y0, p_cf_in_wf = cls.__getInitialData()
        
        var_x = cls.__getAlipMatA('x', t) @ X0
        var_y = cls.__getAlipMatA('y', t) @ Y0

        ref_p_com_in_cf = np.vstack(( var_x[0,0], var_y[0,0], H ))
        ref_p_com_in_wf = ref_p_com_in_cf + p_cf_in_wf
        
        #預測的下一步的初始角動量
        pdc_Ly_com_in_cf_1T = var_x[1,0]
        pdc_Lx_com_in_cf_1T = var_y[1,0]
        
        ref_xy_swTOcom_in_cf_T = np.vstack(( 
            ( des_Ly_com_in_cf_2T - cosh(w*T)*pdc_Ly_com_in_cf_1T ) / ( m*H*w*sinh(w*T) ),
            ( des_Lx_com_in_cf_2T - cosh(w*T)*pdc_Lx_com_in_cf_1T ) / -( m*H*w*sinh(w*T) )
        ))

        
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

    @staticmethod
    def __getInitialData():
        return 
    
    @staticmethod
    def __getAlipMatA(axis:str, t:float) -> np.ndarray:
        """理想ALIP動態矩陣"""
        m = Config.MASS
        H = Config.IDEAL_Z_COM_IN_WF
        w = Config.OMEGA
        
        if axis == 'x':
            return np.array([
                [         cosh(w*t), sinh(w*t)/(m*H*w) ], 
                [ m*H*w * sinh(w*t), cosh(w*t)         ]
            ])
        elif axis == 'y':
            return np.array([
                [          cosh(w*t), -sinh(w*t)/(m*H*w) ],
                [ -m*H*w * sinh(w*t),  cosh(w*t)         ]
            ])


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

