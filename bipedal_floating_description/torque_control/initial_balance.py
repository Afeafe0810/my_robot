#================ import library ========================#
import numpy as np; np.set_printoptions(precision=5)

#================ import library ========================#
from utils.ros_interfaces import RobotModel
from utils.frame_kinermatic import RobotFrame

def balance_ctrl(frame: RobotFrame, robot:RobotModel, jp: np.ndarray) -> np.ndarray:
    """在剛開機狀態直接用關節角度的 單環 來平衡"""
    ref_jp = np.zeros((12,1))
    kp = np.vstack([ 2, 2, 4, 6, 6, 4 ]*2)
    
    tauG_lf, tauG_rf = frame.calculate_gravity(robot, jp, 0, ['lf', 'rf'])
    
    tauG = np.vstack(( tauG_lf, tauG_rf ))
    
    torque = kp * (ref_jp-jp) + tauG
    
    return torque