import numpy as np; np.set_printoptions(precision=2)
#================ import library ========================#
from bipedal_floating_description.motion_planning.utils import Ref, linear_move
from bipedal_floating_description.utils.frame_kinermatic import RobotFrame
from bipedal_floating_description.utils.config import Config
#========================================================#

# state 0, 1
class BipedalBalance:
    def __init__(self):
        self.is_just_started = True
        self.Tk = 0
        
        self.pel0 : np.ndarray = None
        self.lf0  : np.ndarray = None
        self.rf0  : np.ndarray = None

    def plan(self, frame: RobotFrame) -> Ref:
        """左右腳都保持初值, 骨盆線性移動到0.55的高度"""
        
        #剛開始讀取初值
        if self.is_just_started:
            self.pel0 = frame.p_pel_in_wf
            self.lf0 = frame.p_lf_in_wf
            self.rf0 = frame.p_rf_in_wf
            
            self.is_just_started = False
        
        #骨盆高度線性移動
        H = Config.IDEAL_Z_PEL_IN_WF
        z_pel = linear_move(self.Tk, 0, Config.TL_BALANCE, self.pel0[2,0], H)
        
        if self.Tk < Config.TL_BALANCE:
            self.Tk += 1
        
        return Ref(
            pel := np.vstack((self.pel0[:2], z_pel, 0, 0, 0)),
            lf  := np.vstack((self.lf0, 0, 0, 0)),
            rf   = np.vstack((self.rf0, 0, 0, 0)),
            var  = Ref._generate_var_insteadOf_com(pel, lf)
        )