#================ import library ========================#
import numpy as np; np.set_printoptions(precision=5)

#================ import library ========================#
from utils.frame_kinermatic import RobotFrame
from utils.config import Config

class AlipControl:
    """AlipControl 類別負責支撐腳腳踝用ALIP的狀態控制"""
    
    def __init__(self):
        # #(k-1)的輸入
        # self.u_p_lf : dict[str, float] = {
        #     'x' : 0.,
        #     'y' : 0.
        # }
        # self.u_p_rf : dict[str, float] = {
        #     'x' : 0.,
        #     'y' : 0.
        # }
        
        # #(k-1)的估測值
        # self.var_e_p_lf : dict[str, np.ndarray] = {
        #     'x': np.zeros((2,1)),
        #     'y': np.zeros((2,1))
        # }
        # self.var_e_p_rf : dict[str, np.ndarray] = {
        #     'x': np.zeros((2,1)),
        #     'y': np.zeros((2,1))
        # }
        
        # #(k-1)的量測值
        # self.var_p_lf : dict[str, np.ndarray] = {
        #     'x': np.zeros((2,1)),
        #     'y': np.zeros((2,1))
        # }
        # self.var_p_rf : dict[str, np.ndarray] = {
        #     'x': np.zeros((2,1)),
        #     'y': np.zeros((2,1))
        # }
        pass
    
    def ctrl(self, frame:RobotFrame, stance: list[str], stance_past: list[str], ref_var: dict[str, np.ndarray]) -> np.ndarray:
        """回傳支撐腳腳踝扭矩"""
        cf, sf = stance
        #TODO 這邊改用估測控制
        # if stance != stance_past:
        #     self.update_initialValue(stance)
            
        #==========量測的狀態變數==========#
        var_cf = frame.get_alipVar(stance)
        
        #==========過去的變數==========#
        # u_p = {'lf': self.u_p_lf, 'rf': self.u_p_rf}
        # var_e_p = {'lf': self.var_e_p_lf, 'rf': self.var_e_p_rf}
        # var_p = {'lf': self.var_p_lf, 'rf': self.var_p_rf}
        
        # u_p_cf, var_e_p_cf, var_p_cf = u_p[cf], var_e_p[cf], var_p[cf]
        
        #==========離散的狀態矩陣==========#
        matA = {
            'x': np.array([
                [ 1,      0.00247],
                [ 0.8832, 1      ]
            ]),
            
            'y': np.array([
                [  1,     -0.00247],
                [ -0.8832, 1      ]
            ])
        }
        
        matB = {
            'x': np.vstack(( 0, 0.01 )),
            'y': np.vstack(( 0, 0.01 )),
        }
        
        matK = {
            'x': np.array([ [ 150, 15.0198] ]),
            'y': np.array([ [-150, 15     ] ])
        }

        # matL = {
        #     'x': np.array([
        #         [ 0.1390, 0.0025],
        #         [ 0.8832, 0.2803]
        #     ]),
            
        #     'y': np.array([
        #         [  0.1288, -0.0026 ],
        #         [ -0.8832,  0.1480 ]
        #     ])
        # }
        
        # matK = {
        #     'x': np.array([[290.3274,15.0198]])*0.5,
        #     'y': np.array([[-177.0596,9.6014]])*0.15
        # }
        
        matL = {
            'x': np.array([[0.1390,0.0025],[0.8832,0.2803]]),
            
            'y': np.array([[0.1288,-0.0026],[-0.8832,0.1480]])
        }
        
        
        #==========估測器補償==========#
        # var_e_cf = {
        #     'x': matA['x'] @ var_e_p_cf['x'] + matB['x'] * u_p_cf['x'] + matL['x'] @ (var_p_cf['x'] - var_e_p_cf['x']),
        #     'y': matA['y'] @ var_e_p_cf['y'] + matB['y'] * u_p_cf['y'] + matL['y'] @ (var_p_cf['y'] - var_e_p_cf['y']),
        # }
        
        #==========全狀態回授==========#
        u_cf : dict[str, np.ndarray] = {
            'y': -matK['x'] @ ( var_cf['x'] - ref_var['x'] ), #腳踝pitch控制x方向
            'x': -matK['y'] @ ( var_cf['y'] - ref_var['y'] ), #腳踝row控制y方向
        }
        
        u_cf['x'] = u_cf['x'].clip(-Config.ANKLE_X_LIMIT, Config.ANKLE_X_LIMIT) #飽和
        u_cf['y'] = u_cf['y'].clip(-Config.ANKLE_Y_LIMIT, Config.ANKLE_Y_LIMIT) #飽和

        #要補角動量切換！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

        # if self.stance_past == 0 and self.stance == 1:
        #     self.mea_y_L[1,0] = copy.deepcopy(self.mea_y_past_R[1,0])

        

        # if stance != stance_past:
            # u_cf['x'] = u_cf['y'] = 0

        # ALIP計算的u指關節對軀體的扭矩, pub的torque是從骨盆往下建, 扭矩方向相反
        torque_ankle_cf = - np.vstack(( u_cf['y'], u_cf['x'] ))
        
        #==========更新值==========#
        # var_e_p[cf].update(var_e_cf)
        # var_p[cf].update(var_cf)

        return torque_ankle_cf

    # def update_initialValue(self, stance):
    #     cf, sf = stance
    #     #==========過去的變數==========#
    #     u_p = {'lf': self.u_p_lf, 'rf': self.u_p_rf}
    #     var_e_p = {'lf': self.var_e_p_lf, 'rf': self.var_e_p_rf}
    #     var_p = {'lf': self.var_p_lf, 'rf': self.var_p_rf}
        
    #     #切換瞬間扭矩舊值為0
    #     u_p[cf].update({ key: 0.0 for key in u_p})
        
    #     #切換瞬間量測的角動量一樣
    #     var_p[cf]['x'][1] = var_p[sf]['x'][1]
    #     var_p[cf]['y'][1] = var_p[sf]['y'][1]
        
    #     #切換瞬間估測值代入量測值
    #     var_e_p[cf].update(var_p[cf])
            

