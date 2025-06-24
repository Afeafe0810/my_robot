from std_msgs.msg import Float64MultiArray 
import numpy as np; np.set_printoptions(precision=2)
from numpy.typing import NDArray
import pandas as pd
from copy import deepcopy
from math import cos, sin
from scipy.spatial.transform import Rotation as R
import pink
import pinocchio as pin
from itertools import accumulate
#================ import other code =====================#
from bipedal_floating_description.utils.config import Config
from bipedal_floating_description.utils.robot_model import RobotModel
from bipedal_floating_description.utils.signal_process import Dsp
#========================================================#
def zyx_to_geometry(ay: float, az: float) -> NDArray:
    """
    將以 zyx 方法的歐拉角微分 [dax,day,daz] 轉成幾何角動量(座標和輸入的旋轉矩陣的基底座標相同)
    
    原理:  
    az 繞 z1, 方向為 [0, 0, 1]  
    ay 繞 y2, 方向為 Rz1的第二column  
    ax 繞 x3, 方向為 Ry2*Rz1的第一column
    """
    daz_to_w = np.array([0, 0, 1])
    day_to_w = np.array([-sin(az), cos(az), 0])
    dax_to_w = np.array([cos(az)*cos(ay), sin(az)*cos(ay), -sin(ay)])
    
    return np.column_stack([dax_to_w, day_to_w, daz_to_w])

def rotationMat_to_euler(rotationMat: NDArray) -> NDArray:
    """將旋轉矩陣以 zyx 方法的歐拉角回傳 [ax, ay, az]的順序"""
    return R.from_matrix(rotationMat).as_euler('zyx', degrees = False)[::-1]

def rotationMat_of_axis(axe: str, theta: float) -> NDArray:
    """給定x, y, z的旋轉軸方向, 得到 cos, sin 旋轉矩陣"""
    match axe:
        case 'x':
            axe_vec = np.array([1, 0, 0])
        case 'y':
            axe_vec = np.array([0, 1, 0])
        case 'z':
            axe_vec = np.array([0, 0, 1])
            
    return R.from_rotvec(axe_vec * theta).as_matrix()

class PinkFrame:
    """
    PinkFrame 的座標和機器人的 config 的 base 為基準, 而我們輸入的 config 來自於 meshrobot,  
    因此 pf 為平行於骨盆的 base frame
    
    方法是透過 pink 的 get_transform_frame_to_world(), 可以得到對 pf 的 HMT 矩陣
    
    """
    def __init__(self, config: pink.Configuration, robot: RobotModel, jp: NDArray):
        get = lambda name: self._get_OneFrame_in_pf(config, name)
        get_p = lambda name: self._get_OneFrame_in_pf(config, name, 'p')
        
        # 骨盆
        self.p_pel, self.r_from_pel = get("pelvis_link")
        
        # 質心
        self.p_com = robot.bipedal_from_pel.com(jp)
        
        # 左腳連桿
        self.p_hipLower_L, self.r_from_hipLower_L = get("l_hip_yaw_1")
        self.p_hipUpper_L, self.r_from_hipUpper_L = get("l_hip_pitch_1")
        self.p_thigh_L,    self.r_from_thigh_L = get("l_thigh_1")
        self.p_shank_L,    self.r_from_shank_L = get("l_shank_1")
        self.p_ankle_L,    self.r_from_ankleY_L = get("l_ankle_1")
        _,                 self.r_from_ankleX_L = get("l_foot_1")
        self.p_lf,         self.r_from_lf = get("l_foot")
        
        # 右腳連桿
        self.p_hipLower_R, self.r_from_hipLower_R = get("r_hip_yaw_1")
        self.p_hipUpper_R, self.r_from_hipUpper_R = get("r_hip_pitch_1")
        self.p_thigh_R,    self.r_from_thigh_R = get("r_thigh_1")
        self.p_shank_R,    self.r_from_shank_R = get("r_shank_1")
        self.p_ankle_R,    self.r_from_ankleY_R = get("r_ankle_1")
        _,                 self.r_from_ankleX_R = get("r_foot_1")
        self.p_rf,         self.r_from_rf = get("r_foot")
        
        # 左關節
        self.p_jhipX_L = get_p("L_Hip_Roll")
        self.p_jhipY_L = get_p("L_Hip_Pitch")
        self.p_jhipZ_L = get_p("L_Hip_Yaw")
        self.p_jkneeY_L = get_p("L_Knee_Pitch")
        self.p_jankleY_L = get_p("L_Ankle_Pitch")
        self.p_jankleX_L = get_p("L_Ankle_Roll")
        
        # 右關節
        self.p_jhipX_R = get_p("R_Hip_Roll")
        self.p_jhipY_R = get_p("R_Hip_Pitch")
        self.p_jhipZ_R = get_p("R_Hip_Yaw")
        self.p_jkneeY_R = get_p("R_Knee_Pitch")
        self.p_jankleY_R = get_p("R_Ankle_Pitch")
        self.p_jankleX_R = get_p("R_Ankle_Roll")
        
        # 主要端末的姿態
        self.a_pel = rotationMat_to_euler(self.r_from_pel)
        self.a_lf = rotationMat_to_euler(self.r_from_lf)
        self.a_rf = rotationMat_to_euler(self.r_from_rf)
        
        
    @staticmethod
    def _get_OneFrame_in_pf(config: pink.Configuration, framename: str, mode: str = None)->tuple[NDArray, NDArray]:
        se3_htm = config.get_transform_frame_to_world(framename)
        
        match mode:
            case 'p':
                return se3_htm.translation
            case 'r':
                return se3_htm.rotation
            case _:
                return se3_htm.translation, se3_htm.rotation
        
class WorldFrame:
    def __init__(self, pf: PinkFrame, p_base_in_wf: NDArray, r_base_to_wf: NDArray):
        p_pf_in_wf, r_pf_to_wf = p_base_in_wf, r_base_to_wf
        
        def get_p(p_in_pf: np.ndarray) -> np.ndarray: 
            return r_pf_to_wf @ p_in_pf + p_pf_in_wf
        
        def get_r(r_to_pf: np.ndarray) -> np.ndarray:
            return r_pf_to_wf @ r_to_pf
        
        # 骨盆與質心
        _p_pel = get_p(pf.p_pel)
        self.p_com_in_wf = get_p(pf.p_com)
        
        # 左腳link位置
        self.p_hipLower_L = get_p(pf.p_hipLower_L)
        self.p_hipUpper_L = get_p(pf.p_hipUpper_L)
        self.p_thigh_L = get_p(pf.p_thigh_L)
        self.p_shank_L = get_p(pf.p_shank_L)
        self.p_ankle_L = get_p(pf.p_ankle_L)
        _p_lf = get_p(pf.p_lf)
        
        # 右腳link位置
        self.p_hipLower_R = get_p(pf.p_hipLower_R)
        self.p_hipUpper_R = get_p(pf.p_hipUpper_R)
        self.p_thigh_R = get_p(pf.p_thigh_R)
        self.p_shank_R = get_p(pf.p_shank_R)
        self.p_ankle_R = get_p(pf.p_ankle_R)
        _p_rf = get_p(pf.p_rf)
        
        # 主要端點的旋轉矩陣
        self.r_from_pel = get_r(pf.r_from_pel)
        self.r_from_lf = get_r(pf.r_from_lf)
        self.r_from_rf = get_r(pf.r_from_rf)
        
        # 主要的端點(進行濾波)
        self.p_pel_in_wf = Dsp.FILTER_P_PEL_IN_WF.filt(_p_pel)
        self.p_lf_in_wf = Dsp.FILTER_P_LF_IN_WF.filt(_p_lf)
        self.p_rf_in_wf = Dsp.FILTER_P_RF_IN_WF.filt(_p_rf)
        
        
class Jacobian:
    def __init__(self):
        pass
    
class AllFrame:
    def __init__(self):
        self.wf = WorldFrame()
        self.pf = PinkFrame()
        self.jacobian = Jacobian()
        
    
class RobotFrame:
    def __init__(self):        
        pass
    
    #=======================對外的接口=================================#
    def updateFrame(self, robot: RobotModel, config: pink.Configuration, p_base_in_wf: np.ndarray, r_base_to_wf: np.ndarray, jp: np.ndarray):
        '''更新所有frame下的座標'''
        
        #透過機器人的meshcat的config來更新pink frame下的座標
        self.__update_pfFrame(config, robot, jp)
        
        #透過pink的座標, 以及訂閱到的base_in_wf, 可以用運動學得到其他部位in_wf (事後發現, 其實pink frame就是base)
        self.__update_wfFrame(p_base_in_wf, r_base_to_wf)
        
        #得到link間的旋轉矩陣
        self.__update_linkRotMat(jp)
        self.__update_axisVec()
        
        #更新com速度和角動量
        self.__update_com_VandL()
        
        self.eularToGeo ={
            'lf': self.__eularToGeometry(self.pa_lf_in_pf[3:]),
            'rf': self.__eularToGeometry(self.pa_rf_in_pf[3:])
        }
        
        print(self.get_jacobian_of_cfTOLhipyaw())
        print(self.get_jacobian_of_cfTORhipyaw())
        print(self.get_jacobian_of_cfTOLthigh())
        print(self.get_jacobian_of_cfTORthigh())
    
    def get_posture(self) -> tuple[np.ndarray] :
        
        pa_lfTOpel_in_pf = self.pa_pel_in_pf - self.pa_lf_in_pf #骨盆中心相對於左腳
        pa_rfTOpel_in_pf = self.pa_pel_in_pf - self.pa_rf_in_pf #骨盆中心相對於右腳

        return pa_lfTOpel_in_pf, pa_rfTOpel_in_pf
 
    def get_jacobian(self):
        Jp1_L = np.cross( self.axis_1L_in_pf, (self.p_lf_in_pf - self.p_jLhipX_in_pf ), axis = 0 )
        Jp2_L = np.cross( self.axis_2L_in_pf, (self.p_lf_in_pf - self.p_jLhipZ_in_pf ), axis = 0 )
        Jp3_L = np.cross( self.axis_3L_in_pf, (self.p_lf_in_pf - self.p_jLhipY_in_pf ), axis = 0 )
        Jp4_L = np.cross( self.axis_4L_in_pf, (self.p_lf_in_pf - self.p_jLkneeY_in_pf), axis = 0 )
        Jp5_L = np.cross( self.axis_5L_in_pf, (self.p_lf_in_pf - self.p_jLankleY_in_pf ), axis = 0 )
        Jp6_L = np.cross( self.axis_6L_in_pf, (self.p_lf_in_pf - self.p_jLankleX_in_pf ), axis = 0 )
        
        Jp_L = np.hstack(( Jp1_L, Jp2_L, Jp3_L, Jp4_L, Jp5_L, Jp6_L ))
        Ja_L = np.hstack(( 
            self.axis_1L_in_pf, self.axis_2L_in_pf, self.axis_3L_in_pf,
            self.axis_4L_in_pf, self.axis_5L_in_pf, self.axis_6L_in_pf
        ))
        
        JL = np.vstack(( Jp_L, Ja_L ))
        
        Jp1_R = np.cross( self.axis_1R_in_pf, (self.p_rf_in_pf - self.p_jRhipX_in_pf ), axis = 0 )
        Jp2_R = np.cross( self.axis_2R_in_pf, (self.p_rf_in_pf - self.p_jRhipZ_in_pf ), axis = 0 )
        Jp3_R = np.cross( self.axis_3R_in_pf, (self.p_rf_in_pf - self.p_jRhipY_in_pf ), axis = 0 )
        Jp4_R = np.cross( self.axis_4R_in_pf, (self.p_rf_in_pf - self.p_jRkneeY_in_pf), axis = 0 )
        Jp5_R = np.cross( self.axis_5R_in_pf, (self.p_rf_in_pf - self.p_jRankleY_in_pf ), axis = 0 )
        Jp6_R = np.cross( self.axis_6R_in_pf, (self.p_rf_in_pf - self.p_jRankleX_in_pf ), axis = 0 )
        
        Jp_R = np.hstack(( Jp1_R, Jp2_R, Jp3_R, Jp4_R, Jp5_R, Jp6_R ))
        Ja_R = np.hstack(( 
            self.axis_1R_in_pf, self.axis_2R_in_pf, self.axis_3R_in_pf,
            self.axis_4R_in_pf, self.axis_5R_in_pf, self.axis_6R_in_pf
        ))
        
        JR = np.vstack(( Jp_R, Ja_R ))
        
        return JL, JR
    
    def get_jacobian_of_cfTOLhipyaw(self):
        Jp1_L = np.cross( self.axis_6L_in_pf, (self.p_LhipZ_in_pf - self.p_jLankleX_in_pf), axis = 0 )
        Jp2_L = np.cross( self.axis_5L_in_pf, (self.p_LhipZ_in_pf - self.p_jLankleY_in_pf), axis = 0 )
        Jp3_L = np.cross( self.axis_4L_in_pf, (self.p_LhipZ_in_pf - self.p_jLkneeY_in_pf), axis = 0 )
        Jp4_L = np.cross( self.axis_3L_in_pf, (self.p_LhipZ_in_pf - self.p_jLhipY_in_pf), axis = 0 )
        
        Jp_L = np.column_stack(( Jp1_L, Jp2_L, Jp3_L, Jp4_L))
        Ja_L = np.column_stack(( 
            self.axis_6L_in_pf, self.axis_5L_in_pf, self.axis_4L_in_pf, self.axis_3L_in_pf
        ))
        
        JL = np.vstack(( Jp_L, Ja_L ))
        
        
        Jp1_R = np.cross( self.axis_6R_in_pf, (self.p_LhipZ_in_pf - self.p_jRankleX_in_pf), axis = 0 )
        Jp2_R = np.cross( self.axis_5R_in_pf, (self.p_LhipZ_in_pf - self.p_jRankleY_in_pf), axis = 0 )
        Jp3_R = np.cross( self.axis_4R_in_pf, (self.p_LhipZ_in_pf - self.p_jRkneeY_in_pf), axis = 0 )
        Jp4_R = np.cross( self.axis_3R_in_pf, (self.p_LhipZ_in_pf - self.p_jRhipY_in_pf), axis = 0 )
        Jp5_R = np.cross( self.axis_2R_in_pf, (self.p_LhipZ_in_pf - self.p_jRhipZ_in_pf), axis = 0 )
        Jp6_R = np.cross( self.axis_1R_in_pf, (self.p_LhipZ_in_pf - self.p_jRhipX_in_pf), axis = 0 )
        
        Jp_R = np.column_stack(( Jp1_R, Jp2_R, Jp3_R, Jp4_R, Jp5_R, Jp6_R ))
        Ja_R = np.column_stack(( 
            self.axis_6R_in_pf, self.axis_5R_in_pf, self.axis_4R_in_pf,
            self.axis_3R_in_pf, self.axis_2R_in_pf, self.axis_1R_in_pf
        ))
        
        JR = np.vstack(( Jp_R, Ja_R ))
        
        return JL, JR
    
    def get_jacobian_of_cfTOLthigh(self):
        thigh = self.p_LhipY_in_pf
        Jp1_L = np.cross( self.axis_6L_in_pf, (thigh - self.p_jLankleX_in_pf), axis = 0 )
        Jp2_L = np.cross( self.axis_5L_in_pf, (thigh - self.p_jLankleY_in_pf), axis = 0 )
        Jp3_L = np.cross( self.axis_4L_in_pf, (thigh - self.p_jLkneeY_in_pf), axis = 0 )
        
        Jp_L = np.column_stack(( Jp1_L, Jp2_L, Jp3_L))
        Ja_L = np.column_stack(( 
            self.axis_6L_in_pf, self.axis_5L_in_pf, self.axis_4L_in_pf
        ))
        
        JL = np.vstack(( Jp_L, Ja_L ))
        
        
        Jp1_R = np.cross( self.axis_6R_in_pf, (thigh - self.p_jRankleX_in_pf), axis = 0 )
        Jp2_R = np.cross( self.axis_5R_in_pf, (thigh - self.p_jRankleY_in_pf), axis = 0 )
        Jp3_R = np.cross( self.axis_4R_in_pf, (thigh - self.p_jRkneeY_in_pf), axis = 0 )
        Jp4_R = np.cross( self.axis_3R_in_pf, (thigh - self.p_jRhipY_in_pf), axis = 0 )
        Jp5_R = np.cross( self.axis_2R_in_pf, (thigh - self.p_jRhipZ_in_pf), axis = 0 )
        Jp6_R = np.cross( self.axis_1R_in_pf, (thigh - self.p_jRhipX_in_pf), axis = 0 )
        
        Jp7_R = np.cross( self.axis_1L_in_pf, (thigh - self.p_jLhipX_in_pf), axis = 0 )
        Jp8_R = np.cross( self.axis_2L_in_pf, (thigh - self.p_jLhipZ_in_pf), axis = 0 )
        Jp9_R = np.cross( self.axis_3L_in_pf, (thigh - self.p_jLhipY_in_pf), axis = 0 )
        
        Jp_R = np.column_stack(( Jp1_R, Jp2_R, Jp3_R, Jp4_R, Jp5_R, Jp6_R, Jp7_R, Jp8_R, Jp9_R ))
        Ja_R = np.column_stack(( 
            self.axis_6R_in_pf, self.axis_5R_in_pf, self.axis_4R_in_pf,
            self.axis_3R_in_pf, self.axis_2R_in_pf, self.axis_1R_in_pf,
            self.axis_1L_in_pf, self.axis_2L_in_pf, self.axis_3L_in_pf
        ))
        
        JR = np.vstack(( Jp_R, Ja_R ))
        
        return JL, JR
    
    def get_jacobian_of_cfTORhipyaw(self):
        Jp1_L = np.cross( self.axis_6L_in_pf, (self.p_RhipZ_in_pf - self.p_jLankleX_in_pf), axis = 0 )
        Jp2_L = np.cross( self.axis_5L_in_pf, (self.p_RhipZ_in_pf - self.p_jLankleY_in_pf), axis = 0 )
        Jp3_L = np.cross( self.axis_4L_in_pf, (self.p_RhipZ_in_pf - self.p_jLkneeY_in_pf), axis = 0 )
        Jp4_L = np.cross( self.axis_3L_in_pf, (self.p_RhipZ_in_pf - self.p_jLhipY_in_pf), axis = 0 )
        Jp5_L = np.cross( self.axis_2L_in_pf, (self.p_RhipZ_in_pf - self.p_jLhipZ_in_pf), axis = 0 )
        Jp6_L = np.cross( self.axis_1L_in_pf, (self.p_RhipZ_in_pf - self.p_jLhipX_in_pf), axis = 0 )
        
        Jp_L = np.column_stack(( Jp1_L, Jp2_L, Jp3_L, Jp4_L, Jp5_L, Jp6_L ))
        Ja_L = np.column_stack(( 
            self.axis_6L_in_pf, self.axis_5L_in_pf, self.axis_4L_in_pf,
            self.axis_3L_in_pf, self.axis_2L_in_pf, self.axis_1L_in_pf
        ))
        
        JL = np.vstack(( Jp_L, Ja_L ))
        
        
        Jp1_R = np.cross( self.axis_6R_in_pf, (self.p_RhipZ_in_pf - self.p_jRankleX_in_pf), axis = 0 )
        Jp2_R = np.cross( self.axis_5R_in_pf, (self.p_RhipZ_in_pf - self.p_jRankleY_in_pf), axis = 0 )
        Jp3_R = np.cross( self.axis_4R_in_pf, (self.p_RhipZ_in_pf - self.p_jRkneeY_in_pf), axis = 0 )
        Jp4_R = np.cross( self.axis_3R_in_pf, (self.p_RhipZ_in_pf - self.p_jRhipY_in_pf), axis = 0 )
        
        Jp_R = np.column_stack(( Jp1_R, Jp2_R, Jp3_R, Jp4_R))
        Ja_R = np.column_stack(( 
            self.axis_6R_in_pf, self.axis_5R_in_pf, self.axis_4R_in_pf, self.axis_3R_in_pf
        ))
        
        JR = np.vstack(( Jp_R, Ja_R ))        
        return JL, JR
    
    def get_jacobian_of_cfTORthigh(self):
        thigh = self.p_RhipY_in_pf
        
        Jp1_L = np.cross( self.axis_6L_in_pf, (thigh - self.p_jLankleX_in_pf), axis = 0 )
        Jp2_L = np.cross( self.axis_5L_in_pf, (thigh - self.p_jLankleY_in_pf), axis = 0 )
        Jp3_L = np.cross( self.axis_4L_in_pf, (thigh - self.p_jLkneeY_in_pf), axis = 0 )
        Jp4_L = np.cross( self.axis_3L_in_pf, (thigh - self.p_jLhipY_in_pf), axis = 0 )
        Jp5_L = np.cross( self.axis_2L_in_pf, (thigh - self.p_jLhipZ_in_pf), axis = 0 )
        Jp6_L = np.cross( self.axis_1L_in_pf, (thigh - self.p_jLhipX_in_pf), axis = 0 )
        
        Jp7_L = np.cross( self.axis_1R_in_pf, (thigh - self.p_jRhipX_in_pf), axis = 0 )
        Jp8_L = np.cross( self.axis_2R_in_pf, (thigh - self.p_jRhipZ_in_pf), axis = 0 )
        Jp9_L = np.cross( self.axis_3R_in_pf, (thigh - self.p_jRhipY_in_pf), axis = 0 )
        
        Jp_L = np.column_stack(( Jp1_L, Jp2_L, Jp3_L, Jp4_L, Jp5_L, Jp6_L, Jp7_L, Jp8_L, Jp9_L ))
        Ja_L = np.column_stack(( 
            self.axis_6L_in_pf, self.axis_5L_in_pf, self.axis_4L_in_pf,
            self.axis_3L_in_pf, self.axis_2L_in_pf, self.axis_1L_in_pf,
            self.axis_1R_in_pf, self.axis_2R_in_pf, self.axis_3R_in_pf
        ))
        
        JL = np.vstack(( Jp_L, Ja_L ))
        
        Jp1_R = np.cross( self.axis_6R_in_pf, (thigh - self.p_jRankleX_in_pf), axis = 0 )
        Jp2_R = np.cross( self.axis_5R_in_pf, (thigh - self.p_jRankleY_in_pf), axis = 0 )
        Jp3_R = np.cross( self.axis_4R_in_pf, (thigh - self.p_jRkneeY_in_pf), axis = 0 )
        
        Jp_R = np.column_stack(( Jp1_R, Jp2_R, Jp3_R))
        Ja_R = np.column_stack(( 
            self.axis_6R_in_pf, self.axis_5R_in_pf, self.axis_4R_in_pf
        ))
        
        JR = np.vstack(( Jp_R, Ja_R ))
        
        return JL, JR
    
    
    @property
    def p_ft_in_wf(self):
        return {
            'lf': self.p_lf_in_wf,
            'rf': self.p_rf_in_wf
        }
    
    def get_alipVar(self, stance: list[str]) -> dict[str, np.ndarray]:
        """ retrurn alip_var = {'x': [x, Ly], 'y': [y, Lx]}"""
        #HACK 之後可能要改用cf
        cf, sf = stance
        
        p_ftTocom_in_wf = {
            'lf': self.p_com_in_wf - self.p_lf_in_wf,
            'rf': self.p_com_in_wf - self.p_rf_in_wf
        }
        L_com_in_ft = {
            'lf': self.L_com_in_lf,
            'rf': self.L_com_in_rf,
        }
        var = {
            'x': np.vstack(( p_ftTocom_in_wf[cf][0], L_com_in_ft[cf]['y'])),
            'y': np.vstack(( p_ftTocom_in_wf[cf][1], L_com_in_ft[cf]['x'])),
        }
        return deepcopy(var)
     
    @staticmethod
    def rotMat_to_euler(r_to_frame: np.ndarray) -> np.ndarray:
        """ 回傳a_in_frame, 以roll, pitch, yaw的順序(x,y,z), 以column vector"""
        return np.vstack((
            R.from_matrix(r_to_frame).as_euler('zyx', degrees=False)[::-1]
        ))
        
    def to_csv(self, records: pd.DataFrame, stance: list[str]):
        
        var = self.get_alipVar(stance)
        
        this_record = pd.DataFrame([{
            'com_x': self.p_com_in_wf[0,0],
            'com_y': self.p_com_in_wf[1,0],
            'com_z': self.p_com_in_wf[2,0],

            'lf_x': self.p_lf_in_wf[0,0],
            'lf_y': self.p_lf_in_wf[1,0],
            'lf_z': self.p_lf_in_wf[2,0],

            'rf_x': self.p_rf_in_wf[0,0],
            'rf_y': self.p_rf_in_wf[1,0],
            'rf_z': self.p_rf_in_wf[2,0],

            'x': var['x'][0,0],
            'y': var['y'][0,0],
            
            'Ly': var['x'][1, 0],
            'Lx': var['y'][1, 0],
            
            'pel_x': self.p_pel_in_wf[0,0],
            'pel_y': self.p_pel_in_wf[1,0],
            'pel_z': self.p_pel_in_wf[2,0],
        }])
        
        new_records = pd.concat([records, this_record], ignore_index=True)
        
        records.to_csv("real_measure.csv")
        
        return new_records
    #=======================封裝主要的部份================================#
    
    def __update_pfFrame(self, config: pink.Configuration, robot: RobotModel, jp: np.ndarray):
        '''可以得到各部位在pink frame下的資訊'''
        
        #各連桿的質心
        self.p_pel_in_pf,    self.r_pel_to_pf    = self.__getOneInPf(config, "pelvis_link")
        self.p_LhipX_in_pf,  self.r_LhipX_to_pf  = self.__getOneInPf(config, "l_hip_yaw_1") #確認沒錯
        self.p_LhipZ_in_pf,  self.r_LhipZ_to_pf  = self.__getOneInPf(config, "l_hip_pitch_1")
        self.p_LhipY_in_pf,  self.r_LhipY_to_pf  = self.__getOneInPf(config, "l_thigh_1")
        self.p_LkneeY_in_pf, self.r_LkneeY_to_pf = self.__getOneInPf(config, "l_shank_1")
        self.p_LankY_in_pf,  self.r_LankY_to_pf  = self.__getOneInPf(config, "l_ankle_1")
        self.p_LankX_in_pf,  self.r_LankX_to_pf  = self.__getOneInPf(config, "l_foot_1")
        self.p_lf_in_pf,     self.r_lf_to_pf     = self.__getOneInPf(config, "l_foot")
        self.p_RhipX_in_pf,  self.r_RhipX_to_pf  = self.__getOneInPf(config, "r_hip_yaw_1")
        self.p_RhipZ_in_pf,  self.r_RhipZ_to_pf  = self.__getOneInPf(config, "r_hip_pitch_1")
        self.p_RhipY_in_pf,  self.r_RhipY_to_pf  = self.__getOneInPf(config, "r_thigh_1")
        self.p_RkneeY_in_pf, self.r_RkneeY_to_pf = self.__getOneInPf(config, "r_shank_1")
        self.p_RankY_in_pf,  self.r_RankY_to_pf  = self.__getOneInPf(config, "r_ankle_1")
        self.p_RankX_in_pf,  self.r_RankX_to_pf  = self.__getOneInPf(config, "r_foot_1")
        self.p_rf_in_pf,     self.r_rf_to_pf     = self.__getOneInPf(config, "r_foot")
        
        #各關節的位置
        self.p_jLhipX_in_pf, self.r_jLhipX_to_pf = self.__getOneInPf(config, "L_Hip_Roll")
        self.p_jLhipY_in_pf, self.r_jLhipY_to_pf = self.__getOneInPf(config, "L_Hip_Pitch")
        self.p_jLhipZ_in_pf, self.r_jLhipZ_to_pf = self.__getOneInPf(config, "L_Hip_Yaw")
        self.p_jLkneeY_in_pf, self.r_jLkneeY_to_pf = self.__getOneInPf(config, "L_Knee_Pitch")
        self.p_jLankleY_in_pf, self.r_jLankleY_to_pf = self.__getOneInPf(config, "L_Ankle_Pitch")
        self.p_jLankleX_in_pf, self.r_jLankleX_to_pf = self.__getOneInPf(config, "L_Ankle_Roll")
        
        self.p_jRhipX_in_pf, self.r_jRhipX_to_pf = self.__getOneInPf(config, "R_Hip_Roll")
        self.p_jRhipY_in_pf, self.r_jRhipY_to_pf = self.__getOneInPf(config, "R_Hip_Pitch")
        self.p_jRhipZ_in_pf, self.r_jRhipZ_to_pf = self.__getOneInPf(config, "R_Hip_Yaw")
        self.p_jRkneeY_in_pf, self.r_jRkneeY_to_pf = self.__getOneInPf(config, "R_Knee_Pitch")
        self.p_jRankleY_in_pf, self.r_jRankleY_to_pf = self.__getOneInPf(config, "R_Ankle_Pitch")
        self.p_jRankleX_in_pf, self.r_jRankleX_to_pf = self.__getOneInPf(config, "R_Ankle_Roll")
        
        #整個機器人的質心位置
        self.p_com_in_pf = robot.bipedal_from_pel.com(jp)
        
        self.pa_pel_in_pf = np.vstack(( self.p_pel_in_pf, self.rotMat_to_euler(self.r_pel_to_pf) ))
        self.pa_lf_in_pf  = np.vstack(( self.p_lf_in_pf , self.rotMat_to_euler(self.r_lf_to_pf)  ))
        self.pa_rf_in_pf  = np.vstack(( self.p_rf_in_pf , self.rotMat_to_euler(self.r_rf_to_pf)  ))

    
    def __update_wfFrame(self, p_base_in_wf: np.ndarray, r_base_to_wf: np.ndarray):
        '''得到各點在world frame下的資訊'''
        
        p_pf_in_wf, r_pf_to_wf = p_base_in_wf, r_base_to_wf
        
        def place_in_pfToWf(p_in_pf: np.ndarray) -> np.ndarray: 
            return r_pf_to_wf @ p_in_pf + p_pf_in_wf
        
        def rotat_to_pfToWf(r_to_pf: np.ndarray) -> np.ndarray:
            return r_pf_to_wf @ r_to_pf
        
        _p_pel_in_wf = place_in_pfToWf(self.p_pel_in_pf)
        self.p_com_in_wf = place_in_pfToWf(self.p_com_in_pf)
        self.p_LhipX_in_wf = place_in_pfToWf(self.p_LhipX_in_pf) 
        self.p_LhipZ_in_wf = place_in_pfToWf(self.p_LhipZ_in_pf) 
        self.p_LhipY_in_wf = place_in_pfToWf(self.p_LhipY_in_pf) 
        self.p_LkneeY_in_wf = place_in_pfToWf(self.p_LkneeY_in_pf)
        self.p_LankY_in_wf = place_in_pfToWf(self.p_LankY_in_pf) 
        self.p_LankX_in_wf = place_in_pfToWf(self.p_LankX_in_pf) 
        _p_lf_in_wf = place_in_pfToWf(self.p_lf_in_pf)    
        self.p_RhipX_in_wf = place_in_pfToWf(self.p_RhipX_in_pf) 
        self.p_RhipZ_in_wf = place_in_pfToWf(self.p_RhipZ_in_pf) 
        self.p_RhipY_in_wf = place_in_pfToWf(self.p_RhipY_in_pf) 
        self.p_RkneeY_in_wf = place_in_pfToWf(self.p_RkneeY_in_pf)
        self.p_RankY_in_wf = place_in_pfToWf(self.p_RankY_in_pf) 
        self.p_RankX_in_wf = place_in_pfToWf(self.p_RankX_in_pf) 
        _p_rf_in_wf = place_in_pfToWf(self.p_rf_in_pf)    
        
        self.r_pel_to_wf = rotat_to_pfToWf(self.r_pel_to_pf)
        self.r_lf_to_wf = rotat_to_pfToWf(self.r_lf_to_pf)
        self.r_rf_to_wf = rotat_to_pfToWf(self.r_rf_to_pf)
        
        self.p_pel_in_wf = Dsp.FILTER_P_PEL_IN_WF.filt(_p_pel_in_wf)
        self.p_lf_in_wf = Dsp.FILTER_P_LF_IN_WF.filt(_p_lf_in_wf)
        self.p_rf_in_wf = Dsp.FILTER_P_RF_IN_WF.filt(_p_rf_in_wf)
        

    def __update_linkRotMat(self, jps: np.ndarray):
        '''利用旋轉軸的向量得到link間的旋轉矩陣'''
        axes = ['x', 'z', 'y', 'y', 'y', 'x']
        
        self.r_1_to_0_L, self.r_2_to_1_L, self.r_3_to_2_L, self.r_4_to_3_L, self.r_5_to_4_L, self.r_6_to_5_L = [
            self.__get_axis_rotMat(axis, jp) for axis, jp in zip( axes, jps[:6,0] ) 
        ]
        
        self.r_1_to_0_R, self.r_2_to_1_R, self.r_3_to_2_R, self.r_4_to_3_R, self.r_5_to_4_R, self.r_6_to_5_R = [
            self.__get_axis_rotMat(axis, jp) for axis, jp in zip( axes, jps[6:,0] ) 
        ]

    def __update_axisVec(self):
        '''得到各關節軸的單位向量'''
        
        #==========骨盆到腳底各軸的方向==========#
        axes = ['x', 'z', 'y', 'y', 'y', 'x']
        
        #==========軸方向的映射==========#
        mapping = {
            'x': np.vstack(( 1, 0, 0 )),
            'y': np.vstack(( 0, 1, 0 )),
            'z': np.vstack(( 0, 0, 1 )),
        }
        vec_axes = [ mapping[axis] for axis in axes ]
        
        #==========各link到base的旋轉矩陣==========#
        r_n_to_0_L = list(accumulate(
            [ np.eye(3), self.r_1_to_0_L, self.r_2_to_1_L, self.r_3_to_2_L, self.r_4_to_3_L, self.r_5_to_4_L ],
            func = np.matmul
        ))
        r_n_to_0_R = list(accumulate(
            [ np.eye(3), self.r_1_to_0_R, self.r_2_to_1_R, self.r_3_to_2_R, self.r_4_to_3_R, self.r_5_to_4_R ],
            func = np.matmul
        ))
        
        #==========旋轉軸的向量==========#
        (
            self.axis_1L_in_pf, self.axis_2L_in_pf, self.axis_3L_in_pf,
            self.axis_4L_in_pf, self.axis_5L_in_pf, self.axis_6L_in_pf
            
        ) = [ r_n_to_0_L[i] @ vec_axes[i] for i in range(6) ]
        
        (
            self.axis_1R_in_pf, self.axis_2R_in_pf, self.axis_3R_in_pf,
            self.axis_4R_in_pf, self.axis_5R_in_pf, self.axis_6R_in_pf
            
        ) = [ r_n_to_0_R[i] @ vec_axes[i] for i in range(6) ]
    
    def __update_com_VandL(self):
        m = Config.MASS
        H = Config.IDEAL_Z_COM_IN_WF
        
        _v_com_in_wf = Dsp.DIFFTER_P_COM_IN_WF.diff(self.p_com_in_wf)
        self.v_com_in_wf = Dsp.FILTER_V_COM_IN_WF.filt(_v_com_in_wf)
        
        p0_ftTocom_in_wf = {
            'lf': self.p_com_in_wf - self.p_lf_in_wf,
            'rf': self.p_com_in_wf - self.p_rf_in_wf
        }
        #學長直接用0.45, 要注意一下到底要不要改
        self.L_com_in_lf = {
            'y':   m * H * self.v_com_in_wf[0,0],
            'x': - m * H * self.v_com_in_wf[1,0],
        }
        self.L_com_in_rf = {
            'y':   m * H * self.v_com_in_wf[0,0],
            'x': - m * H * self.v_com_in_wf[1,0],
        }
        
    #========================toolbox================================#
    @staticmethod
    # def __getOneInPf(config: pink.Configuration, link: str):
    #     htm = config.get_transform_frame_to_world(link)
    #     p_in_pf = np.reshape( htm.translation, (3,1) )
    #     r_to_pf = np.reshape( htm.rotation, (3,3) )
    #     return p_in_pf, r_to_pf 
    
    @staticmethod
    def __get_axis_rotMat(axis: str, theta:float)->np.ndarray:
        vec_theta = [theta, 0, 0] if axis == 'x' else\
                    [0, theta, 0] if axis == 'y' else\
                    [0, 0, theta] if axis == 'z' else None
                    
        return R.from_rotvec(vec_theta).as_matrix()
        
    @staticmethod
    def __eularToGeometry(angle: np.ndarray):
        _, ay, az = angle.flatten()
        return np.array([
            [ cos(ay)*cos(az), -sin(az), 0 ],
            [ cos(ay)*sin(az),  cos(az), 0 ],
            [        -sin(ay),        0, 1 ],
        ])