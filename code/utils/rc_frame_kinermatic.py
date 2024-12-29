from std_msgs.msg import Float64MultiArray 
import numpy as np; np.set_printoptions(precision=2)
from copy import deepcopy
from math import cos, sin
from scipy.spatial.transform import Rotation as R
import pink
import pinocchio as pin
#================ import other code =====================#
from utils.config import Config
from utils.ros_interfaces import ROSInterfaces
#========================================================#

class RobotFrame:
    def __init__(self):
        self.P_PEL_IN_BASE = np.vstack(( 0, 0, 0.598 ))
        self.r_pf_to_pel = np.identity(3)
        
        self.p_base_in_wf: np.ndarray = None
        self.r_base_to_wf: np.ndarray = None
        
        pass
    
    def updateFrame(self, ros: ROSInterfaces, config: pink.Configuration, p_base_in_wf: np.ndarray, r_base_to_wf: np.ndarray, jp: np.ndarray):
        self.__update_pfFrame(config, ros, jp)
        self.__update_wfFrame(p_base_in_wf, r_base_to_wf)
        
        self.eularToGeo ={
            'lf': self.__eularToGeometry(self.pa_lf_in_pf[3:]),
            'rf': self.__eularToGeometry(self.pa_rf_in_pf[3:])
        }
        
        return (
            (self.p_pel_in_pf,    self.r_pel_to_pf   ),
            (self.p_LhipX_in_pf,  self.r_LhipX_to_pf ),
            (self.p_LhipZ_in_pf,  self.r_LhipZ_to_pf ),
            (self.p_LhipY_in_pf,  self.r_LhipY_to_pf ),
            (self.p_LkneeY_in_pf, self.r_LkneeY_to_pf),
            (self.p_LankY_in_pf,  self.r_LankY_to_pf ),
            (self.p_LankX_in_pf,  self.r_LankX_to_pf ),
            (self.p_lf_in_pf,     self.r_lf_to_pf    ),
            (self.p_RhipX_in_pf,  self.r_RhipX_to_pf ),
            (self.p_RhipZ_in_pf,  self.r_RhipZ_to_pf ),
            (self.p_RhipY_in_pf,  self.r_RhipY_to_pf ),
            (self.p_RkneeY_in_pf, self.r_RkneeY_to_pf),
            (self.p_RankY_in_pf,  self.r_RankY_to_pf ),
            (self.p_RankX_in_pf,  self.r_RankX_to_pf ),
            (self.p_rf_in_pf,     self.r_rf_to_pf    ),

            self.pa_pel_in_pf, self.pa_lf_in_pf, self.pa_rf_in_pf,
            
            self.p_pel_in_wf, self.r_pel_to_wf,
            
            self.p_com_in_wf   ,
            self.p_LhipX_in_wf ,
            self.p_LhipZ_in_wf ,
            self.p_LhipY_in_wf ,
            self.p_LkneeY_in_wf,
            self.p_LankY_in_wf ,
            self.p_LankX_in_wf ,
            self.p_lf_in_wf    ,
            self.p_RhipX_in_wf ,
            self.p_RhipZ_in_wf ,
            self.p_RhipY_in_wf ,
            self.p_RkneeY_in_wf,
            self.p_RankY_in_wf ,
            self.p_RankX_in_wf ,
            self.p_rf_in_wf    ,
            
            self.r_lf_to_wf    ,
            self.r_rf_to_wf,
        )
    
    @staticmethod
    def get_posture(pa_pel_in_pf, pa_lf_in_pf, pa_rf_in_pf):
        
        pa_lfTOpel_in_pf = pa_pel_in_pf - pa_lf_in_pf #骨盆中心相對於左腳
        pa_rfTOpel_in_pf = pa_pel_in_pf - pa_rf_in_pf #骨盆中心相對於右腳

        return pa_lfTOpel_in_pf, pa_rfTOpel_in_pf
    
    def __update_pfFrame(self, config: pink.Configuration, ros: ROSInterfaces, jp: np.ndarray):
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
        
        self.pa_pel_in_pf = np.vstack(( self.p_pel_in_pf, self.__rotToEuler(self.r_pel_to_pf) ))
        self.pa_lf_in_pf  = np.vstack(( self.p_lf_in_pf , self.__rotToEuler(self.r_lf_to_pf)  ))
        self.pa_rf_in_pf  = np.vstack(( self.p_rf_in_pf , self.__rotToEuler(self.r_rf_to_pf)  ))
              
    def __update_wfFrame(self, p_base_in_wf: np.ndarray, r_base_to_wf: np.ndarray):
        self.p_pel_in_wf, self.r_pel_to_wf = self.__get_pelInWf(p_base_in_wf, r_base_to_wf)
        
        r_pf_to_wf = self.r_pel_to_wf @ self.r_pel_to_pf.T
        place_in_pfToWf = lambda p_in_pf: r_pf_to_wf @ (p_in_pf - self.p_pel_in_pf) + self.p_pel_in_wf
        rotat_to_pfToWf = lambda r_to_pf: r_pf_to_wf @ r_to_pf
        
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
        
        self.r_lf_to_wf = rotat_to_pfToWf(self.r_lf_to_pf)
        self.r_rf_to_wf = rotat_to_pfToWf(self.r_rf_to_pf)
 
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
        p_com_in_pf = np.reshape(ros.bipedal_floating_data.com[0],(3,1))
        return p_com_in_pf
           
    def __get_pelInWf(self, p_base_in_wf, r_base_to_wf):
        return(
            r_base_to_wf @ self.P_PEL_IN_BASE + p_base_in_wf,
            deepcopy(r_base_to_wf)
        )
        
    @staticmethod
    def __rotToEuler(r_to_frame: np.ndarray)->np.ndarray:
        """ 回傳a_in_frame, 以roll, pitch, yaw的順序(x,y,z), 以column vector"""
        return np.vstack((
            R.from_matrix(r_to_frame).as_euler('zyx', degrees=False)[::-1]
        ))
    
    @staticmethod
    def __eularToGeometry(angle: np.ndarray):
        _, ay, az = angle.flatten()
        return np.array([
            [ cos(ay)*cos(az), -sin(az), 0 ],
            [ cos(ay)*sin(az),  cos(az), 0 ],
            [        -sin(ay),        0, 1 ],
        ])