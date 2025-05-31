import numpy as np; np.set_printoptions(precision=5)
from numpy.typing import NDArray
import pandas as pd
#================ import library ========================#
from utils.ros_interfaces import ROSInterfaces as ROS
from utils.frame_kinermatic import RobotFrame
from utils.config import Config

class _Abstract_Alip_EstimateBias:
    def __init__(self, A: NDArray, B: NDArray, K: NDArray, L: NDArray, L_bias: NDArray, limit: float):
        self.is_ctrl_first_time = True
        
        #狀態矩陣
        self.A  = A
        self.B  = B
        self.K  = K
        self.L  = L
        self.L_bias = L_bias
        self.limit = limit
        
        #估測狀態
        # self.var_e: NDArray = np.vstack((0.02, 0))
        self.bias_e: float = 0
    
    def ctrl(self, frame: RobotFrame, stance: list[str], stance_past: list[str], mea_pel: float, ref_var: NDArray) -> float:
        """回傳支撐腳腳踝扭矩"""
        if stance != stance_past or self.is_ctrl_first_time:
            self.is_ctrl_first_time = False
            self.var_e = np.vstack((mea_pel, 0))

        #==========全狀態回授(飽和)==========# HACK 用估測的骨盆位置來進行回授
        _u = -self.K @ (self.var_e + np.vstack((self.bias_e, 0))- ref_var) 
        u = np.clip(_u, -self.limit, self.limit)
        
        #==========估測回授==========#
        err_e = mea_pel - self.var_e[0, 0] - self.bias_e
        self.var_e = self.A @ self.var_e + self.B * u + self.L * err_e
        self.bias_e = self.bias_e + self.L_bias * err_e
        
        return -u

class _Abstract_Alip_NoEstimate:
    def __init__(self, A: NDArray, B: NDArray, K: NDArray, L: NDArray, L_bias: NDArray, limit: float):
        self.is_ctrl_first_time = True
        
        #狀態矩陣
        self.A  = A
        self.B  = B
        self.K  = K
        self.limit = limit
    
    def ctrl(self, frame: RobotFrame, stance: list[str], stance_past: list[str], var: NDArray, ref_var: NDArray) -> float:
        """回傳支撐腳腳踝扭矩"""
        if stance != stance_past or self.is_ctrl_first_time:
            self.is_ctrl_first_time = False
            self.var_e = var

        #==========全狀態回授(飽和)==========#
        _u = -self.K @ (var - ref_var) 
        u = np.clip(_u, -self.limit, self.limit)
        
        return -u
    
# class AlipX(_Abstract_Alip_EstimateBias):
#     def __init__(self):
#         super().__init__(
#             A = np.array([[1, 0.00247], [0.8832, 1]]),
#             B = np.vstack((0, 0.01)),
#             K = np.array([[150,15.0198]]),
#             L = np.vstack((1.656737, 24.448707)),
#             L_bias = -1.238861,
#             limit = Config.ANKLE_AY_LIMIT
#         )
    
#     def ctrl(self, frame: RobotFrame, stance: list[str], stance_past: list[str], mea_pel: float, ref_var: NDArray) -> float:
#         ref_var = np.vstack((0.005, 0))
#         mea_pel = frame.p_pel_in_wf[0, 0] - frame.p_lf_in_wf[0, 0]
        
#         if self.is_ctrl_first_time:
#             self.is_ctrl_first_time = False
#             self.var_e = np.vstack((mea_pel, 0))
            
#         output = super().ctrl(frame, stance, stance_past, mea_pel, ref_var)
#         return output

class AlipX(_Abstract_Alip_NoEstimate):
    def __init__(self):
        super().__init__(
            A = np.array([[1, 0.00247], [0.8832, 1]]),
            B = np.vstack((0, 0.01)),
            K = np.array([[150,15.0198]]),
            L = np.vstack((1.656737, 24.448707)),
            L_bias = -1.238861,
            limit = Config.ANKLE_AY_LIMIT
        )
            
# class AlipY(_Abstract_Alip_NoEstimate):
#     def __init__(self):
#         super().__init__(
#             A = np.array([[1, -0.00247],[-0.8832, 1]]),
#             B = np.vstack((0, 0.01)),
#             # K = np.array([[-150, 15]]),
#             # K = np.array([-177.059600, 9.601400]),
#             # K = np.array([-185.618059, 9.804615]),
#             K = np.array([-127.045296, 6.185515])*0.8,
#             L = np.vstack((3.274146, -40.792358)),
#             L_bias = -2.730338
#         )
#     def ctrl(self, frame: RobotFrame, stance: list[str], stance_past: list[str], var: NDArray, ref_var: NDArray, limit: float) -> float:
#         ref_var = np.vstack((0, 0))
#         u = super().ctrl(stance, stance_past, var, ref_var, limit)
#         print(pd.Series({
#             "com": var[0, 0],
#             "pel": frame.p_pel_in_wf[1, 0]-frame.p_lf_in_wf[1, 0],
#             "ref_com": ref_var[0, 0]
#         }))
#         return u
        
class AlipY1(_Abstract_Alip_EstimateBias):
    def __init__(self):
        super().__init__(
            A = np.array([[1, -0.00247],[-0.8832, 1]]),
            B = np.vstack((0, 0.01)),
            K = np.array([[-150, 15]]),
            # K = np.array([-127.045296, 6.185515]),
            # K = np.array([-177.059600, 9.601400]),
            # L = np.vstack((0.680529, -11.882289)),
            # L_bias = -0.395041
            L = np.vstack((0.680529, -11.882289)),
            L_bias = -0.395041,
            limit = Config.ANKLE_AX_LIMIT
        )
        self.bias_e = 0.02

    def ctrl(self, frame: RobotFrame, stance: list[str], stance_past: list[str], mea_pel: float, ref_var: NDArray) -> float:
        ref_var = np.vstack((0.005, 0))
        mea_pel = frame.p_pel_in_wf[1, 0] - frame.p_lf_in_wf[1, 0]
        
        if self.is_ctrl_first_time:
            self.is_ctrl_first_time = False
            self.var_e = np.vstack((-0.04, 0))
            
        data = pd.Series({
            'com_e': self.var_e[0, 0],
            'com': frame.p_com_in_wf[0, 0] - frame.p_lf_in_wf[0, 0],
            'pel_e': self.var_e[0, 0] + self.bias_e,
            'pel': mea_pel,
            'ref_var': ref_var[0, 0]
        })
        print(data)
        ROS.publishers.com_e.publish([data['com_e']])
        ROS.publishers.com.publish([data["com"]])
        ROS.publishers.pel_e.publish([data['pel_e']])
        ROS.publishers.pel.publish([data["pel"]])
        
        output = super().ctrl(frame, stance, stance_past, mea_pel, ref_var)
        return output