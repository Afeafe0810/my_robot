#================ import library ========================#
import numpy as np; np.set_printoptions(precision=5)
import pandas as pd
#================ import library ========================#
from utils.robot_model import RobotModel
from utils.frame_kinermatic import RobotFrame
from motion_planning import Ref
from utils.config import Config

from torque_control.knee_control import KneeLoop
from torque_control.pd_control import balance_ctrl, cf_anklePD_Ax, cf_anklePD, cf_anklePD_Ax2
from torque_control.alip_control import AlipX, AlipY, AlipY1
from torque_control.ankle_control import anklePD_ctrl


# TODO 碰撞偵測：正常的力 - 碰撞的力，再經過低通濾波器
class TorqueControl:
    """TorqueControl 類別負責處理機器人扭矩對state的pattern matching邏輯。"""
    
    def __init__(self):
        self.knee = KneeLoop()
        #self.alip = AlipControl()
        self.alipx = AlipX()
        self.alipy = AlipY()
        self.alipy1 = AlipY1()
        self.alipT = 0
        
    def update_torque(self, frame: RobotFrame, robot: RobotModel, ref: Ref, state: float,
                      stance: list[str], stance_past: list[str], is_firmly: dict[str, bool], jp: np.ndarray, jv: np.ndarray) -> np.ndarray:
        
        cf, sf = stance
        
        match state:
            case 0:
                return balance_ctrl(frame, robot, jp)
            case 1:
                #雙腳膝蓋
                torque_knee = self.knee.ctrl(ref, frame, robot, jp, jv, state, stance, is_firmly)
                torque_ankle_ay = self.alipx.ctrl(frame, stance, stance_past, frame.get_alipVar(stance)['x'], ref.var['x'], Config.ANKLE_AY_LIMIT)
                torque_ankle_ax = cf_anklePD_Ax(frame, robot, jp, jv)
                #雙腳腳踝
                torque_ankle = {
                    sf : anklePD_ctrl(frame, sf),
                    cf : np.vstack((torque_ankle_ay, torque_ankle_ax))
                    # cf : cf_anklePD(frame, robot, jp, jv)
                }
                
                return np.vstack(( torque_knee['lf'], torque_ankle['lf'], torque_knee['rf'], torque_ankle['rf'] ))
            case 2:
                #雙腳膝蓋
                torque_knee = self.knee.ctrl(ref, frame, robot, jp, jv, state, stance, is_firmly)
                torque_ankle_ay = self.alipx.ctrl(stance, stance_past, frame.get_alipVar(stance)['x'], ref.var['x'], Config.ANKLE_AY_LIMIT)
                torque_ankle_ax = cf_anklePD_Ax2(frame, robot, ref.ax, jp, jv, ref.var['y'][0, 0])
                #雙腳腳踝
                torque_ankle = {
                    sf : anklePD_ctrl(frame, sf),
                    cf : np.vstack((torque_ankle_ay, torque_ankle_ax))
                    # cf : cf_anklePD(frame, robot, jp, jv)
                }
                
                return np.vstack(( torque_knee['lf'], torque_ankle['lf'], torque_knee['rf'], torque_ankle['rf'] ))
            case 3:
                #雙腳膝蓋
                torque_knee = self.knee.ctrl(ref, frame, robot, jp, jv, state, stance, is_firmly)
                torque_ankle_ay = self.alipx.ctrl(stance, stance_past, frame.get_alipVar(stance)['x'], ref.var['x'], Config.ANKLE_AY_LIMIT)
                torque_ankle_ax = self.alipy1.ctrl(frame, stance, stance_past, frame.get_alipVar(stance)['y'], ref.var['y'], Config.ANKLE_AX_LIMIT)
                #雙腳腳踝
                torque_ankle = {
                    sf : anklePD_ctrl(frame, sf),
                    cf : np.vstack((torque_ankle_ay, torque_ankle_ax))
                    # cf : cf_anklePD(frame, robot, jp, jv)
                }
                
                return np.vstack(( torque_knee['lf'], torque_ankle['lf'], torque_knee['rf'], torque_ankle['rf'] ))
            # case 30:
            #     #雙腳膝蓋
            #     torque_knee = self.knee.ctrl(ref, frame, robot, jp, jv, state, stance, is_firmly)
                
            #     #雙腳腳踝
            #     torque_ankle = {
            #         sf : anklePD_ctrl(frame, sf),
            #         cf : self.alip.ctrl(frame, stance, stance_past, ref.var)
            #     }
                
            #     return np.vstack(( torque_knee['lf'], torque_ankle['lf'], torque_knee['rf'], torque_ankle['rf'] ))