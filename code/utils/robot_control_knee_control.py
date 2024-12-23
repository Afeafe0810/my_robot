#================ import library ========================#
from std_msgs.msg import Float64MultiArray 
import pinocchio as pin

import numpy as np; np.set_printoptions(precision=2)

from sys import argv
from os.path import dirname, join, abspath
import copy

def outterloop():
    pass

def endErr_to_endVel(self):
    ref_pa_pel_in_wf = copy.deepcopy(self.PX_ref) #wf
    ref_pa_lf_in_wf  = copy.deepcopy(self.LX_ref) #wf
    ref_pa_rf_in_wf  = copy.deepcopy(self.RX_ref) #wf
    pa_pel_in_pf = copy.deepcopy(self.PX) #pf
    pa_lf_in_pf  = copy.deepcopy(self.LX) #pf
    pa_rf_in_pf  = copy.deepcopy(self.RX) #pf
    
    #========求相對骨盆的向量========#
    ref_pa_pelTOlf_in_pf = ref_pa_lf_in_wf -ref_pa_pel_in_wf
    ref_pa_pelTOrf_in_pf = ref_pa_rf_in_wf -ref_pa_pel_in_wf
    
    pa_pelTOlf_in_pf = pa_lf_in_pf -pa_pel_in_pf
    pa_pelTOrf_in_pf = pa_rf_in_pf -pa_pel_in_pf
    
    #========經加法器算誤差========#
    err_pa_pelTOlf_in_pf = ref_pa_pelTOlf_in_pf - pa_pelTOlf_in_pf
    err_pa_pelTOrf_in_pf = ref_pa_pelTOrf_in_pf - pa_pelTOrf_in_pf

    #========經P gain作為微分========#
    derr_pa_pelTOlf_in_pf = 20 * err_pa_pelTOlf_in_pf
    derr_pa_pelTOrf_in_pf = 20 * err_pa_pelTOrf_in_pf
    
    #========歐拉角速度轉幾何角速度========#
    w_pelTOlf_in_pf = self.L_Body_transfer @ derr_pa_pelTOlf_in_pf[3:]
    w_pelTOrf_in_pf = self.R_Body_transfer @ derr_pa_pelTOrf_in_pf[3:]
    
    vw_pelTOlf_in_pf = np.vstack(( derr_pa_pelTOlf_in_pf[:3], w_pelTOlf_in_pf ))
    vw_pelTOrf_in_pf = np.vstack(( derr_pa_pelTOrf_in_pf[:3], w_pelTOrf_in_pf ))

    return vw_pelTOlf_in_pf, vw_pelTOrf_in_pf

def endVel_to_jv(Le_2,Re_2,jv_f,stance_type,state, JLL, JRR):
    state = copy.deepcopy(state)
    stance = copy.deepcopy(stance_type)
    
    cf, sf = ('lf','rf') if stance == 0 else \
             ('rf','lf') # if stance ==1, 2
    
    endVel = {'lf': Le_2, 'rf': Re_2}
    jv = {'lf' : jv_f[:6], 'rf':jv_f[6:]}
    jv_ankle = {'lf': jv['lf'][-2:], 'rf': jv['rf'][-2:]}
    J = {
        'lf': JLL,
        'rf': JRR
    }
    if state == 0:
        cmd_jv = {
            'lf': np.linalg.pinv(J['lf']) @ endVel['lf'],
            'rf': np.linalg.pinv(J['rf']) @ endVel['rf']
        }
        
    if state == 1:#真雙支撐
        #===========================支撐腳膝上四關節: 控骨盆z, axyz==================================#
        #===========================擺動腳膝上四關節: 控落點xyz, az==================================#
        ctrlVel = {
            cf: endVel[cf][2:],
            sf: endVel[sf][[0,1,2,5]]
        }
        J_ankle_to_ctrlVel = {
            cf: J[cf][2:, 4:],
            sf: J[sf][[0,1,2,-1], 4:]
        }
        
        J_knee_to_ctrlVel = {
            cf: J[cf][2:, :4],
            sf: J[sf][[0,1,2,-1], :4]
        }
        
        cmd_jv_knee = {
             cf: np.linalg.pinv(J_knee_to_ctrlVel[cf]) @ ( ctrlVel[cf] - J_ankle_to_ctrlVel[cf] @ jv_ankle[cf] ),
             sf: np.linalg.pinv(J_knee_to_ctrlVel[sf]) @ ( ctrlVel[sf] - J_ankle_to_ctrlVel[sf] @ jv_ankle[sf] )
        }
        cmd_jv = {
            cf: np.vstack(( cmd_jv_knee[cf], 0, 0 )),
            sf: np.vstack(( cmd_jv_knee[sf], 0, 0 ))
        }
    
    return cmd_jv['lf'], cmd_jv['rf']
