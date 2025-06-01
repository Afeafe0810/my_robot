import numpy as np; np.set_printoptions(precision=5)
from numpy.typing import NDArray
import pandas as pd
#================ import library ========================#
from utils.ros_interfaces import ROSInterfaces as ROS
from utils.frame_kinermatic import RobotFrame
from utils.config import Config

class AlipX:
    def __init__(self):
        self.is_ctrl_first_time = True

        #狀態矩陣
        self.A  = np.array([[1, 0.00247], [0.8832, 1]])
        self.B  = np.vstack((0, 0.01))
        self.K  = np.array([[150,15.0198]])
        self.L  = np.vstack((1.656737, 24.448707))
        self.L_bias = -1.238861
        self.limit = Config.ANKLE_AY_LIMIT
        
        #估測狀態
        # self.var_e: NDArray = np.vstack((0.02, 0))
        self.bias_e: float = 0
    
    def ctrl(self, frame: RobotFrame, stance: list[str], stance_past: list[str], mea_pel: float, ref_var: NDArray) -> float:
        """回傳支撐腳腳踝扭矩"""
        
        ref_var = np.vstack((-0.02, 0))
        mea_pel = frame.p_pel_in_wf[0, 0] - frame.p_lf_in_wf[0, 0]
        mea_L = frame.L_com_in_lf['y']
        mea_com = frame.p_com_in_wf[0, 0] - frame.p_lf_in_wf[0, 0]
        
        if stance != stance_past or self.is_ctrl_first_time:
            self.is_ctrl_first_time = False
            self.var_e = np.vstack((0, 0))
            self.bias_e = -0.02

        #==========全狀態回授(飽和)==========# HACK 用估測的骨盆位置來進行回授
        _u = -self.K @ (np.vstack((self.var_e[0, 0] + self.bias_e, mea_L)) - ref_var)
        # _u = -self.K @ (self.var_e - ref_var)
        u = np.clip(_u, -self.limit, self.limit)
        
        datax = {
            'com_e': self.var_e[0, 0],
            'com': mea_com,
            'pel_e': self.var_e[0, 0] + self.bias_e,
            'pel': mea_pel
        }
        print(pd.Series(datax))
        #==========估測回授==========#
        err_e = mea_pel - self.var_e[0, 0] - self.bias_e
        self.var_e = self.A @ self.var_e + self.B * u + self.L * err_e
        self.bias_e = self.bias_e + self.L_bias * err_e
        
        return -u
    
    
class AlipY:
    def __init__(self):
        self.is_ctrl_first_time = True
        
        #狀態矩陣
        self.A  = np.array([[1, -0.00247],[-0.8832, 1]])
        self.B  = np.vstack((0, 0.01))
        self.K  = np.array([[-150, 15]])
        self.L  = np.vstack((0.680529, -11.882289))
        self.L_bias = -0.395041
        self.limit = Config.ANKLE_AX_LIMIT
        
        #估測狀態
        # self.var_e: NDArray = np.vstack((0.02, 0))
        self.bias_e: float = 0.02
    
    def ctrl(self, frame: RobotFrame, stance: list[str], stance_past: list[str], mea_pel: float, ref_var: NDArray) -> float:
        """回傳支撐腳腳踝扭矩"""
        ref_var = np.vstack((0.005, 0))
        mea_pel = frame.p_pel_in_wf[1, 0] - frame.p_lf_in_wf[1, 0]
        
        if stance != stance_past or self.is_ctrl_first_time:
            self.is_ctrl_first_time = False
            self.var_e = np.vstack((-0.04, 0))

        #==========全狀態回授(飽和)==========# HACK 用估測的骨盆位置來進行回授
        _u = -self.K @ (self.var_e + np.vstack((self.bias_e, 0))- ref_var) 
        u = np.clip(_u, -self.limit, self.limit)
        
        #==========估測回授==========#
        err_e = mea_pel - self.var_e[0, 0] - self.bias_e
        self.var_e = self.A @ self.var_e + self.B * u + self.L * err_e
        self.bias_e = self.bias_e + self.L_bias * err_e
        
        return -u