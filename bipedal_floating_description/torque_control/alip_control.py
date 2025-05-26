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

class _Abstract_Alip_NoEstimate:
    def __init__(self, A: NDArray, B: NDArray, K: NDArray, L: NDArray, L_bias: NDArray):
        self.is_ctrl_first_time = True
        
        #狀態矩陣
        self.A  = A
        self.B  = B
        self.K  = K
    
    def ctrl(self, stance: list[str], stance_past: list[str], var: NDArray, ref_var: NDArray, limit: float) -> float:
        """回傳支撐腳腳踝扭矩"""
        if stance != stance_past or self.is_ctrl_first_time:
            self.is_ctrl_first_time = False
            self.var_e = var

        #==========全狀態回授(飽和)==========#
        _u = -self.K @ (var - ref_var) 
        u = np.clip(_u, -limit, limit)
        
        return -u
    
class AlipX(_Abstract_Alip_NoEstimate):
    def __init__(self):
        super().__init__(
            A = np.array([[1, 0.00247], [0.8832, 1]]),
            B = np.vstack((0, 0.01)),
            K = np.array([[150,15.0198]]),
            L = np.vstack((1.656737, 24.448707)),
            L_bias = -1.238861
        )
            
class AlipY(_Abstract_Alip_NoEstimate):
    def __init__(self):
        super().__init__(
            A = np.array([[1, -0.00247],[-0.8832, 1]]),
            B = np.vstack((0, 0.01)),
            K = np.array([[-150, 15]]),
            L = np.vstack((3.274146, -40.792358)),
            L_bias = -2.730338
        )
    def ctrl(self, stance: list[str], stance_past: list[str], var: NDArray, ref_var: NDArray, limit: float) -> float:
        u = super().ctrl(stance, stance_past, var, ref_var, limit)
        return u
        
# class AlipY(_Abstract_Alip_EstimateBias):
#     def __init__(self):
#         super().__init__(
#             A = np.array([[1, -0.00247],[-0.8832, 1]]),
#             B = np.vstack((0, 0.01)),
#             K = np.array([[-150, 15]]),
#             L = np.vstack((0.680529, -11.882289)),
#             L_bias = -0.395041
#         )
#     def ctrl(self, stance: list[str], stance_past: list[str], var: NDArray, ref_var: NDArray, limit: float) -> float:
#         u = super().ctrl(stance, stance_past, var, np.zeros((2, 1)), limit)
#         print("com_e:", self.var_e[0, 0])
#         print("bias_e: ", self.bias_e)
#         print("pel_e:", self.var_e[0, 0] + self.bias_e)
#         print("pel:", var[0, 0])
#         print("u_:", u)
#         return u