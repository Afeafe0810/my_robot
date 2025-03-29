#================ import library ========================#
import numpy as np; np.set_printoptions(precision=5)

#================ import library ========================#
from utils.frame_kinermatic import RobotFrame

def anklePD_ctrl(frame: RobotFrame, sf: str):
    """支撐腳腳踝的PD控制"""
    ref_jp = np.zeros((2,1))

    r_ft_to_wf = {
        'lf': frame.r_lf_to_wf,
        'rf': frame.r_rf_to_wf
    }
    ayx_sf_in_wf = frame.rotMat_to_euler(r_ft_to_wf[sf]) [1:]
    #TODO 摩擦力看要不要加
    torque_ankle_sf = 0.1 * ( ref_jp - ayx_sf_in_wf ) # HACK 現在只用P control
    
    return torque_ankle_sf
