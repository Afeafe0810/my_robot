#================ import library ========================#
import numpy as np; np.set_printoptions(precision=5)
from numpy.typing import NDArray
#================ import library ========================#
from utils.frame_kinermatic import RobotFrame
from utils.config import Config

class _AbstractAlipParam:
    def __init__(self):
        #狀態矩陣
        self.A : NDArray
        self.B : NDArray
        self.K : NDArray
        self.L : NDArray
        
        #估測狀態
        self.var_e: NDArray
        
        self.is_ctrl_first_time = True
    
    def ctrl(self, stance: list[str], stance_past: list[str], var: NDArray, ref_var: NDArray, limit: float) -> float:
        """回傳支撐腳腳踝扭矩"""
        cf, sf = stance
        
        if stance != stance_past or self.is_ctrl_first_time:
            self.is_ctrl_first_time = False
            self.var_e = var

        #==========全狀態回授(飽和)==========# HACK 目前沒有用估測回授
        _u = -self.K @ (var - ref_var) 
        u = np.clip(_u, -limit, limit)
        
        #==========更新過去值==========#
        self.var_e = self.A @ self.var_e + self.B * u + self.L @ (var - self.var_e)
        
        return -u
    
class AlipX(_AbstractAlipParam):
    def __init__(self):
        super().__init__()
        self.A = np.array([[1, 0.00247], [0.8832, 1]])
        self.B = np.vstack((0, 0.01))
        self.K = np.array([[290.3274,15.0198]])*0.5
        self.L = np.array([[0.1390,0.0025],[0.8832,0.2803]])

class AlipY(_AbstractAlipParam):
    def __init__(self):
        super().__init__()
        self.A = np.array([[1, -0.00247],[-0.8832, 1]])
        self.B = np.vstack((0, 0.01))
        self.K = np.array([[-177.0596,9.6014]])*0.15
        self.L = np.array([[0.1288,-0.0026],[-0.8832,0.1480]])

class AlipControl:
    def __init__(self):
        self.dir_x = AlipX()
        self.dir_y = AlipY()
    
    def ctrl(self, frame:RobotFrame, stance: list[str], stance_past: list[str], ref_var: dict[str, NDArray]) -> NDArray:
        var = frame.get_alipVar(stance)
        
        tau_ay = self.dir_x.ctrl(stance, stance_past, var['x'], ref_var['x'], Config.ANKLE_AY_LIMIT)
        tau_ax = self.dir_y.ctrl(stance, stance_past, var['y'], ref_var['y'], Config.ANKLE_AX_LIMIT)
        
        return np.vstack((tau_ay, tau_ax))
        
        