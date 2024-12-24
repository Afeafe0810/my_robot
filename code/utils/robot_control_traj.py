import numpy as np; np.set_printoptions(precision=2)

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
    
    return ref_p_pel_in_wf, ref_p_lf_in_wf, ref_p_rf_in_wf
    
def __comMoveTolf(DS_time, DDT):
    ref_pa_lf_in_wf  = np.vstack(( 0,  0.1,  0, 0, 0, 0 ))
    ref_pa_rf_in_wf  = np.vstack(( 0, -0.1,  0, 0, 0, 0 ))
    
    ref_pa_pel_in_wf = np.vstack(( 0, 0.06*DS_time/( 0.5*DDT ), 0.55, 0, 0, 0 )) if 0 < DS_time <= 0.5*DDT else \
                      np.vstack(( 0, 0.06,                     0.55, 0, 0, 0 ))
    return ref_pa_pel_in_wf, ref_pa_lf_in_wf, ref_pa_rf_in_wf
