#================ import library ========================#
import numpy as np; np.set_printoptions(precision=5)

#================ import library ========================#
from utils.robot_model import RobotModel
from utils.frame_kinermatic import RobotFrame
from motion_planning import Ref

from torque_control.knee_control import KneeLoop
from torque_control.initial_balance import balance_ctrl
from torque_control.alip_control import AlipControl
from torque_control.ankle_control import anklePD_ctrl


# TODO 碰撞偵測：正常的力 - 碰撞的力，再經過低通濾波器
class TorqueControl:
    """TorqueControl 類別負責處理機器人扭矩對state的pattern matching邏輯。"""
    
    def __init__(self):
        self.knee = KneeLoop()
        self.alip = AlipControl()
        
    def update_torque(self, frame: RobotFrame, robot: RobotModel, ref: Ref, state: float,
                      stance: list[str], stance_past: list[str], is_firmly: dict[str, bool], jp: np.ndarray, jv: np.ndarray) -> np.ndarray:
        
        cf, sf = stance
        
        match state:
            case 0:
                return balance_ctrl(frame, robot, jp)
            case 1 | 2 | 30:
                #雙腳膝蓋
                torque = self.knee.ctrl(ref, frame, robot, jp, jv, state, stance, is_firmly)
                
                #擺動腳腳踝
                torque[sf][4:6] = anklePD_ctrl(frame, sf)
                
                #支撐腳腳踝
                torque[cf][4:6] = self.alip.ctrl(frame, stance, stance_past, ref.var)
                
                return np.vstack(( torque['lf'], torque['rf'] ))