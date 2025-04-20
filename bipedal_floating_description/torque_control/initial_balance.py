#================ import library ========================#
import numpy as np; np.set_printoptions(precision=5)

#================ import library ========================#
from utils.robot_model import RobotModel
from utils.frame_kinermatic import RobotFrame

def balance_ctrl(frame: RobotFrame, robot:RobotModel, jp: np.ndarray) -> np.ndarray:
    """在剛開機狀態直接用關節角度的 單環 來平衡"""
    ref_jp = np.zeros((12,1))
    kp = np.vstack([ 2, 2, 4, 6, 6, 4 ]*2)
    
    tauG = robot.gravity(jp, 0, ['lf', 'rf'], *frame.get_posture())
    
    torque = kp * (ref_jp-jp) + tauG
    
    return torque