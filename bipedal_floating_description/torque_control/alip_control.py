#================ import library ========================#
import numpy as np; np.set_printoptions(precision=5)
from numpy.typing import NDArray
#================ import library ========================#
from utils.frame_kinermatic import RobotFrame
from utils.config import Config

class _Abstract_Alip_EstimateBias:
    def __init__(self, A: NDArray, B: NDArray, K: NDArray, L: NDArray, L_bias: NDArray):
        self.is_ctrl_first_time = True
        
        #狀態矩陣
        self.A  = A
        self.B  = B
        self.K  = K
        self.L  = L
        self.L_bias = L_bias
        
        #估測狀態
        self.var_e: NDArray = None
        self.bias_e: float = 0.0
    
    def ctrl(self, stance: list[str], stance_past: list[str], var: NDArray, ref_var: NDArray, limit: float) -> float:
        """回傳支撐腳腳踝扭矩"""
        if stance != stance_past or self.is_ctrl_first_time:
            self.is_ctrl_first_time = False
            self.var_e = var

        #==========全狀態回授(飽和)==========#
        _u = -self.K @ (self.var_e - ref_var) 
        u = np.clip(_u, -limit, limit)
        
        #==========估測回授==========#
        err_e = var[0, 0] - self.var_e[0, 0] - self.bias_e
        self.var_e = self.A @ self.var_e + self.B * u + self.L * err_e
        self.bias_e = self.bias_e + self.L_bias * err_e
        
        return -u
    
class _Abstract_AlipY_EstimateBias:
    def __init__(self, A: NDArray, B: NDArray, K: NDArray, L: NDArray, L_bias: NDArray):
        self.is_ctrl_first_time = True
        
        #狀態矩陣
        self.A  = A
        self.B  = B
        self.K  = K
        self.L  = L
        self.L_bias = L_bias
        
        #估測狀態
        self.var_e: NDArray = None
        self.bias_e: float = 0.0
    
    def ctrl(self, stance: list[str], stance_past: list[str], var: NDArray, ref_var: NDArray, limit: float) -> float:
        """回傳支撐腳腳踝扭矩"""
        if stance != stance_past or self.is_ctrl_first_time:
            self.is_ctrl_first_time = False
            self.var_e = var

        #==========全狀態回授==========#
        _u = -self.K @ (var - ref_var) 
        u = np.clip(_u, -limit, limit)
        
        # #==========估測回授==========#
        # err_e = var[0, 0] - self.var_e[0, 0] - self.bias_e
        # self.var_e = self.A @ self.var_e + self.B * u + self.L * err_e
        # self.bias_e = self.bias_e + self.L_bias * err_e
        
        return -u

class AlipX(_Abstract_AlipY_EstimateBias):
    def __init__(self):
        super().__init__(
            A = np.array([[1, 0.00247], [0.8832, 1]]),
            B = np.vstack((0, 0.01)),
            K = np.array([[150,15.0198]]),
            L = np.vstack((1.656737, 24.448707)),
            L_bias = -1.238861
        )
            
class AlipY(_Abstract_AlipY_EstimateBias):
    def __init__(self):
        super().__init__(
            A = np.array([[1, -0.00247],[-0.8832, 1]]),
            B = np.vstack((0, 0.01)),
            # K = np.array([[-177.0596,9.6014]])*0.15,
            K = np.array([[-150, 15]]),
            # K = np.array([-90.890461, 1.593617]),
            L = np.vstack((1.656737, -24.448707)),
            L_bias = -1.238861
        )

# class AlipControl:
#     def __init__(self):
#         self.dir_x = AlipX()
#         self.dir_y = AlipY()
    
#     def ctrl(self, frame:RobotFrame, stance: list[str], stance_past: list[str], ref_var: dict[str, NDArray]) -> NDArray:
#         var = frame.get_alipVar(stance)
        
#         tau_ay = self.dir_x.ctrl(stance, stance_past, var['x'], ref_var['x'], Config.ANKLE_AY_LIMIT)
#         tau_ax = self.dir_y.ctrl(stance, stance_past, var['y'], ref_var['y'], Config.ANKLE_AX_LIMIT)
        
#         return np.vstack((tau_ay, tau_ax))
        
        