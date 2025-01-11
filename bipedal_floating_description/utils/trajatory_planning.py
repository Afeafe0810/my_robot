import numpy as np; np.set_printoptions(precision=2)
from math import cosh, sinh, cos, sin, pi
#================ import library ========================#
from utils.config import Config
from utils.frame_kinermatic import RobotFrame
#========================================================#

class Trajatory:
    def __init__(self):
        
        #state 2, 單腳舉起的時間
        self.leg_lift_time = 0
        
        #state 30, ALIP行走當下時間
        self.alip_time = 0
        
    def plan(self, state):
        if state in [0,1]: #假雙支撐, 真雙支撐
            return self.__bipedalBalanceTraj()
        
        elif state == 2: #骨盆移到支撐腳
            if self.leg_lift_time <= 10 * Config.DDT:
                self.leg_lift_time += Config.TIMER_PERIOD
                print("DS",self.leg_lift_time)
                
            return self.__comMoveTolf(self.leg_lift_time)
        
        elif state == 30: #ALIP規劃
            pass
    
    @staticmethod
    def __bipedalBalanceTraj():
        return {
            'pel': np.vstack(( 0,    0, 0.57, 0, 0, 0 )),
            'lf' : np.vstack(( 0,  0.1,    0, 0, 0, 0 )),
            'rf' : np.vstack(( 0, -0.1,    0, 0, 0, 0 )),
        }
    
    @staticmethod
    def __comMoveTolf(t):
        T = Config.DDT

        #==========線性移動==========#
        linearMove = lambda t, x0, x1, t0, t1:\
            np.clip(x0 + (x1-x0) * (t-t0)/(t1-t0), x0, x1 )
            
        y_pel = linearMove(t, *[0, 0.09], *[0*T, 0.5*T])
        z_sf  = linearMove(t, *[0, 0.05], *[1*T, 1.1*T])
        
        return {
            'pel': np.vstack(( 0, y_pel, 0.55, 0, 0, 0 )),
            'lf' : np.vstack(( 0,   0.1,    0, 0, 0, 0 )),
            'rf' : np.vstack(( 0,  -0.1, z_sf, 0, 0, 0 )),
        }

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

