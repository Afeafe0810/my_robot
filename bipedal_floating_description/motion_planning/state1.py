import numpy as np; np.set_printoptions(precision=2)
#================ import library ========================#
from motion_planning.utils import Ref
from utils.frame_kinermatic import RobotFrame
from utils.config import Config
#========================================================#

# state 0, 1
class BipedalBalance:
    def __init__(self):
        self.is_just_started = True
        self.t = 0.0
        
        self.pel0 : np.ndarray = None
        self.lf0  : np.ndarray = None
        self.rf0  : np.ndarray = None

    def plan(self, frame: RobotFrame) -> Ref:
        """回傳雙腳支撐時的參考值"""
        if self.is_just_started:
            self.pel0, self.lf0, self.rf0 = frame.p_pel_in_wf, frame.p_lf_in_wf, frame.p_rf_in_wf
            self.is_just_started = False
        
        return Ref(
            pel := np.vstack((self.pel0, 0, 0, 0)),
            
            lf  := np.vstack((self.lf0, 0, 0, 0)),
            rf   = np.vstack((self.rf0, 0, 0, 0)),
            var  = Ref._generate_var_insteadOf_com(pel, lf)
        )