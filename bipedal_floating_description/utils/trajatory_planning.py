import numpy as np; np.set_printoptions(precision=2)

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

