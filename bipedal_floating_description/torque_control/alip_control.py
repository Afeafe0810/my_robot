#================ import library ========================#
import numpy as np; np.set_printoptions(precision=5)

#================ import library ========================#
from utils.frame_kinermatic import RobotFrame
from utils.config import Config

class AlipControl:
    """AlipControl 類別負責支撐腳腳踝用ALIP的狀態控制"""
    
    def __init__(self):
        #==========(k-1)的輸入==========#
        # ax方向的扭矩控制側方向y, ay方向的扭矩控制向前x
        # (因為很容易和線方向搞混, 但又不想用row,pitch命名, 所以決定還是改用ax, ay來命名)
        self.u_p_ft = {
            'lf': {'ax': np.zeros((1,1)), 'ay': np.zeros((1,1))},
            'rf': {'ax': np.zeros((1,1)), 'ay': np.zeros((1,1))}
        }
        
        #==========(k-1)的估測值==========#
        self.var_e_p_ft = {
            'lf': {'x': np.zeros((2,1)), 'y': np.zeros((2,1))},
            'rf': {'x': np.zeros((2,1)), 'y': np.zeros((2,1))}
        }
        
        #==========(k-1)的量測值==========#
        self.var_p_ft = {
            'lf': {'x': np.zeros((2,1)), 'y': np.zeros((2,1))},
            'rf': {'x': np.zeros((2,1)), 'y': np.zeros((2,1))}
        }
        
        #==========狀態矩陣==========#
        self.matA = {
            'x': np.array([
                [ 1,      0.00247],
                [ 0.8832, 1      ]
            ]),
            
            'y': np.array([
                [  1,     -0.00247],
                [ -0.8832, 1      ]
            ])
        }
        
        self.matB = {
            'x': np.vstack(( 0, 0.01 )),
            'y': np.vstack(( 0, 0.01 )),
        }
        # self.matK = { #沒有估測時用的K
        #     'x': np.array([ [ 150, 15.0198] ]),
        #     'y': np.array([ [-150, 15     ] ])
        # }
        self.matK = {
            'x': np.array([[290.3274,15.0198]])*0.5,
            'y': np.array([[-177.0596,9.6014]])*0.15
        }
        self.matL = {
            'x': np.array([[0.1390,0.0025],[0.8832,0.2803]]),
            'y': np.array([[0.1288,-0.0026],[-0.8832,0.1480]])
        }
    
    def ctrl(self, frame:RobotFrame, stance: list[str], stance_past: list[str], ref_var: dict[str, np.ndarray]) -> np.ndarray:
        """回傳支撐腳腳踝扭矩"""
        cf, sf = stance
            
        #==========現在量測的狀態變數==========#
        var = frame.get_alipVar(stance)
        
        #==========換腳瞬間過去值的取法==========#
        if stance != stance_past:
            self.set_past_value_for_stance_switch(stance, var)
        
        #==========過去的輸入與狀態變數==========#
        u_p, var_e_p, var_p = self.u_p_ft[cf], self.var_e_p_ft[cf], self.var_p_ft[cf]

        #==========狀態方程式與全狀態回授==========#
        var_e = {
            'x': self.matA['x'] @ var_e_p['x'] + self.matB['x'] * u_p['ay'] + self.matL['x'] @ (var_p['x'] - var_e_p['x']),
            'y': self.matA['y'] @ var_e_p['y'] + self.matB['y'] * u_p['ax'] + self.matL['y'] @ (var_p['y'] - var_e_p['y']),
        }
        
        u : dict[str, np.ndarray] = {
            'ay': -self.matK['x'] @ ( var['x'] - ref_var['x'] ),
            'ax': -self.matK['y'] @ ( var['y'] - ref_var['y'] ),
        }
        
        #==========設定扭矩飽和==========#
        u['ax'] = u['ax'].clip(-Config.ANKLE_AX_LIMIT, Config.ANKLE_AX_LIMIT)
        u['ay'] = u['ay'].clip(-Config.ANKLE_AY_LIMIT, Config.ANKLE_AY_LIMIT)
        
        # ALIP計算的u指的是關節對軀體的扭矩, pub的torque是從骨盆往下建, 扭矩方向相反
        torque_ankle_cf = - np.vstack(( u['ay'], u['ax'] ))
        
        #==========更新過去值==========#
        self.u_p_ft[cf] = u
        self.var_e_p_ft[cf] = var_e
        self.var_p_ft[cf] = var

        return torque_ankle_cf
    
    def set_past_value_for_stance_switch(self, stance: list[str], var: dict[str, np.ndarray]):
        """設定換腳瞬間過去值的取法"""
        
        cf, sf = stance
        #換腳瞬間的「位置」過去值用現在的量測值
        self.var_e_p_ft[cf]['x'][0,0] = self.var_p_ft[cf]['x'][0,0] = var['x'][0,0]
        self.var_e_p_ft[cf]['y'][0,0] = self.var_p_ft[cf]['y'][0,0] = var['y'][0,0]
        
        #換腳瞬間的「角動量」過去值用連續性(左右腳相等)
        self.var_p_ft[cf]['x'][1,0] = self.var_p_ft[sf]['x'][1,0]
        self.var_p_ft[cf]['y'][1,0] = self.var_p_ft[sf]['y'][1,0]
        self.var_e_p_ft[cf]['x'][1,0] = self.var_e_p_ft[sf]['x'][1,0]
        self.var_e_p_ft[cf]['y'][1,0] = self.var_e_p_ft[sf]['y'][1,0]
        
        #換腳瞬間的「扭矩」過去值用連續性(左右腳相等) #HACK 有點猶豫該不該這樣設定還是設成0
        self.u_p_ft[cf] = self.u_p_ft[sf]
        