from std_msgs.msg import Float64MultiArray 
import numpy as np; np.set_printoptions(precision=2)
from copy import deepcopy
from math import cos, sin
from scipy.spatial.transform import Rotation as R
import pink
import pinocchio as pin
from itertools import accumulate
#================ import other code =====================#
from utils.config import Config
from utils.ros_interfaces import ROSInterfaces
from utils.signal_process import Dsp
#========================================================#

class RobotFrame:
    def __init__(self):        
        pass
    
    #=======================對外的接口=================================#
    def updateFrame(self, ros: ROSInterfaces, config: pink.Configuration, p_base_in_wf: np.ndarray, r_base_to_wf: np.ndarray, jp: np.ndarray):
        '''更新所有frame下的座標'''
        
        #透過機器人的meshcat的config來更新pink frame下的座標
        self.__update_pfFrame(config, ros, jp)
        
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
    
    @staticmethod
    def get_posture(pa_pel_in_pf, pa_lf_in_pf, pa_rf_in_pf):
        
        pa_lfTOpel_in_pf = pa_pel_in_pf - pa_lf_in_pf #骨盆中心相對於左腳
        pa_rfTOpel_in_pf = pa_pel_in_pf - pa_rf_in_pf #骨盆中心相對於右腳

        return pa_lfTOpel_in_pf, pa_rfTOpel_in_pf
 
    def left_leg_jacobian(self):
        Jp1_L = np.cross( self.axis_1L_in_pf, (self.p_lf_in_pf - self.p_LhipX_in_pf ), axis = 0 )
        Jp2_L = np.cross( self.axis_2L_in_pf, (self.p_lf_in_pf - self.p_LhipZ_in_pf ), axis = 0 )
        Jp3_L = np.cross( self.axis_3L_in_pf, (self.p_lf_in_pf - self.p_LhipY_in_pf ), axis = 0 )
        Jp4_L = np.cross( self.axis_4L_in_pf, (self.p_lf_in_pf - self.p_LkneeY_in_pf), axis = 0 )
        Jp5_L = np.cross( self.axis_5L_in_pf, (self.p_lf_in_pf - self.p_LankY_in_pf ), axis = 0 )
        Jp6_L = np.cross( self.axis_6L_in_pf, (self.p_lf_in_pf - self.p_LankX_in_pf ), axis = 0 )
        
        Jp_L = np.hstack(( Jp1_L, Jp2_L, Jp3_L, Jp4_L, Jp5_L, Jp6_L ))
        Ja_L = np.hstack(( 
            self.axis_1L_in_pf, self.axis_2L_in_pf, self.axis_3L_in_pf,
            self.axis_4L_in_pf, self.axis_5L_in_pf, self.axis_6L_in_pf
        ))
        
        JL = np.vstack(( Jp_L, Ja_L ))
        
        Jp1_R = np.cross( self.axis_1R_in_pf, (self.p_rf_in_pf - self.p_RhipX_in_pf ), axis = 0 )
        Jp2_R = np.cross( self.axis_2R_in_pf, (self.p_rf_in_pf - self.p_RhipZ_in_pf ), axis = 0 )
        Jp3_R = np.cross( self.axis_3R_in_pf, (self.p_rf_in_pf - self.p_RhipY_in_pf ), axis = 0 )
        Jp4_R = np.cross( self.axis_4R_in_pf, (self.p_rf_in_pf - self.p_RkneeY_in_pf), axis = 0 )
        Jp5_R = np.cross( self.axis_5R_in_pf, (self.p_rf_in_pf - self.p_RankY_in_pf ), axis = 0 )
        Jp6_R = np.cross( self.axis_6R_in_pf, (self.p_rf_in_pf - self.p_RankX_in_pf ), axis = 0 )
        
        Jp_R = np.hstack(( Jp1_R, Jp2_R, Jp3_R, Jp4_R, Jp5_R, Jp6_R ))
        Ja_R = np.hstack(( 
            self.axis_1R_in_pf, self.axis_2R_in_pf, self.axis_3R_in_pf,
            self.axis_4R_in_pf, self.axis_5R_in_pf, self.axis_6R_in_pf
        ))
        
        JR = np.vstack(( Jp_R, Ja_R ))
        
        return JL, JR
    
    def get_alipdata(self, stance):
        """先用wf代替in cf好了"""
        cf, sf = stance
        
        p_ft_in_wf = {
            'lf': self.p_lf_in_wf,
            'rf': self.p_rf_in_wf
        }
        p0_ftTocom_in_wf = {
            'lf': self.p_com_in_wf - self.p_lf_in_wf,
            'rf': self.p_com_in_wf - self.p_rf_in_wf
        }
        L_com_in_ft = {
            'lf': self.L_com_in_lf,
            'rf': self.L_com_in_rf,
        }
        var0 = {
            'x': np.vstack(( p0_ftTocom_in_wf[cf][0], L_com_in_ft[cf]['y'])),
            'y': np.vstack(( p0_ftTocom_in_wf[cf][1], L_com_in_ft[cf]['x'])),
        }
        return list( map( deepcopy, 
            [var0, p0_ftTocom_in_wf, p_ft_in_wf]
        ))
    
    #=======================封裝主要的部份================================#
    
    def __update_pfFrame(self, config: pink.Configuration, ros: ROSInterfaces, jp: np.ndarray):
        '''可以得到各部位在pink frame下的資訊'''
        self.p_pel_in_pf,    self.r_pel_to_pf    = self.__getOneInPf(config, "pelvis_link")
        self.p_LhipX_in_pf,  self.r_LhipX_to_pf  = self.__getOneInPf(config, "l_hip_yaw_1")
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
        
        self.p_com_in_pf = self.__get_comInPf(ros, jp)
        
        self.pa_pel_in_pf = np.vstack(( self.p_pel_in_pf, self.__rotMat_to_euler(self.r_pel_to_pf) ))
        self.pa_lf_in_pf  = np.vstack(( self.p_lf_in_pf , self.__rotMat_to_euler(self.r_lf_to_pf)  ))
        self.pa_rf_in_pf  = np.vstack(( self.p_rf_in_pf , self.__rotMat_to_euler(self.r_rf_to_pf)  ))
              
    def __update_wfFrame(self, p_base_in_wf: np.ndarray, r_base_to_wf: np.ndarray):
        '''得到各點在world frame下的資訊'''
        
        p_pf_in_wf, r_pf_to_wf = p_base_in_wf, r_base_to_wf
        
        place_in_pfToWf = lambda p_in_pf: r_pf_to_wf @ p_in_pf + p_pf_in_wf
        rotat_to_pfToWf = lambda r_to_pf: r_pf_to_wf @ r_to_pf
        
        self.p_pel_in_wf    = place_in_pfToWf(self.p_pel_in_pf)
        self.p_com_in_wf    = place_in_pfToWf(self.p_com_in_pf)
        self.p_LhipX_in_wf  = place_in_pfToWf(self.p_LhipX_in_pf) 
        self.p_LhipZ_in_wf  = place_in_pfToWf(self.p_LhipZ_in_pf) 
        self.p_LhipY_in_wf  = place_in_pfToWf(self.p_LhipY_in_pf) 
        self.p_LkneeY_in_wf = place_in_pfToWf(self.p_LkneeY_in_pf)
        self.p_LankY_in_wf  = place_in_pfToWf(self.p_LankY_in_pf) 
        self.p_LankX_in_wf  = place_in_pfToWf(self.p_LankX_in_pf) 
        self.p_lf_in_wf     = place_in_pfToWf(self.p_lf_in_pf)    
        self.p_RhipX_in_wf  = place_in_pfToWf(self.p_RhipX_in_pf) 
        self.p_RhipZ_in_wf  = place_in_pfToWf(self.p_RhipZ_in_pf) 
        self.p_RhipY_in_wf  = place_in_pfToWf(self.p_RhipY_in_pf) 
        self.p_RkneeY_in_wf = place_in_pfToWf(self.p_RkneeY_in_pf)
        self.p_RankY_in_wf  = place_in_pfToWf(self.p_RankY_in_pf) 
        self.p_RankX_in_wf  = place_in_pfToWf(self.p_RankX_in_pf) 
        self.p_rf_in_wf     = place_in_pfToWf(self.p_rf_in_pf)    
        
        self.r_pel_to_wf = rotat_to_pfToWf(self.r_pel_to_pf)
        self.r_lf_to_wf  = rotat_to_pfToWf(self.r_lf_to_pf)
        self.r_rf_to_wf  = rotat_to_pfToWf(self.r_rf_to_pf)

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
        
        _v_com_in_wf = Dsp.DIFFTER_p_com_in_wf.diff(self.p_com_in_wf)
        self.v_com_in_wf = Dsp.FILTER_v_com_in_wf.filt(_v_com_in_wf)
        
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
    def __getOneInPf(config: pink.Configuration, link: str):
        htm = config.get_transform_frame_to_world(link)
        p_in_pf = np.reshape( htm.translation, (3,1) )
        r_to_pf = np.reshape( htm.rotation, (3,3) )
        return p_in_pf, r_to_pf 
    
    def __get_comInPf(self, ros: ROSInterfaces, jp: np.ndarray):
        
        pin.centerOfMass(
            ros.bipedal_floating_model, ros.bipedal_floating_data, jp
        )
        #TODO: 傳機器人資料就好了，不用傳ros
        p_com_in_pf = np.reshape(ros.bipedal_floating_data.com[0],(3,1))
        return p_com_in_pf
        
    @staticmethod
    def __rotMat_to_euler(r_to_frame: np.ndarray)->np.ndarray:
        """ 回傳a_in_frame, 以roll, pitch, yaw的順序(x,y,z), 以column vector"""
        return np.vstack((
            R.from_matrix(r_to_frame).as_euler('zyx', degrees=False)[::-1]
        ))
    
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