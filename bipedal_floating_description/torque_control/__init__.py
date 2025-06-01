#================ import library ========================#
import numpy as np; np.set_printoptions(precision=5)
import pandas as pd
#================ import library ========================#
from utils.robot_model import RobotModel
from utils.frame_kinermatic import RobotFrame
from motion_planning import Ref
from utils.config import Config

from torque_control.knee_control import KneeLoop
import torque_control.pd_control as PD
from torque_control.alip_control import AlipX, AlipY


# TODO 碰撞偵測：正常的力 - 碰撞的力，再經過低通濾波器
class TorqueControl:
    """TorqueControl 類別負責處理機器人扭矩對state的pattern matching邏輯。"""
    
    def __init__(self):
        self.knee = KneeLoop()
        #self.alip = AlipControl()
        self.alipx = AlipX()
        # self.alipy = AlipY()
        self.alipy = AlipY()
        
    def update_torque(self, frame: RobotFrame, robot: RobotModel, ref: Ref, state: float,
                      stance: list[str], stance_past: list[str], is_firmly: dict[str, bool], jp: np.ndarray, jv: np.ndarray) -> np.ndarray:
        
        cf, sf = stance
        
        match state:
            case 0:
                return PD.initial_balance(frame, robot, jp, jv)
            case 1:
                #雙腳膝蓋
                torque_knee = self.knee.ctrl(ref, frame, robot, jp, jv, state, stance, is_firmly)
                torque_ankle_ay = self.alipx.ctrl(frame, stance, stance_past, frame.get_alipVar(stance)['x'], ref.var['x'])
                torque_ankle_ax = PD.ankle_ax1_cf(frame, robot, jp, jv)
                #雙腳腳踝
                torque_ankle = {
                    sf : PD.ankle_ax_sf(frame, sf),
                    cf : np.vstack((torque_ankle_ay, torque_ankle_ax))
                    # cf : cf_anklePD(frame, robot, jp, jv)
                }
                
                return np.vstack(( torque_knee['lf'], torque_ankle['lf'], torque_knee['rf'], torque_ankle['rf'] ))
            case 2:
                #雙腳膝蓋
                torque_knee = self.knee.ctrl(ref, frame, robot, jp, jv, state, stance, is_firmly)
                torque_ankle_ay = self.alipx.ctrl(frame, stance, stance_past, frame.get_alipVar(stance)['x'], ref.var['x'])
                self.alipx.K  = np.array([[290.3274, 15.0198]])*0.5
                torque_ankle_ax = PD.ankle_ax2_cf(frame, robot, jp, jv, ref.ax)
                #雙腳腳踝
                torque_ankle = {
                    sf : PD.ankle_ax_sf(frame, sf),
                    cf : np.vstack((torque_ankle_ay, torque_ankle_ax))
                    # cf : cf_anklePD(frame, robot, jp, jv)
                }
                
                return np.vstack(( torque_knee['lf'], torque_ankle['lf'], torque_knee['rf'], torque_ankle['rf'] ))
            case 3:
                #雙腳膝蓋
                torque_knee = self.knee.ctrl(ref, frame, robot, jp, jv, state, stance, is_firmly)
                torque_ankle_ay = self.alipx.ctrl(frame, stance, stance_past, frame.get_alipVar(stance)['x'], ref.var['x'])
                self.alipx.K  = np.array([[290.3274, 15.0198]])*0.8
                torque_ankle_ax = self.alipy.ctrl(frame, stance, stance_past, frame.get_alipVar(stance)['y'], ref.var['y'])
                #雙腳腳踝
                torque_ankle = {
                    sf : PD.ankle_ax_sf(frame, sf),
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