from std_msgs.msg import Float64MultiArray 
import numpy as np; np.set_printoptions(precision=2)
from copy import deepcopy
from math import cos, sin
from scipy.spatial.transform import Rotation as R
import pink
#================ import other code =====================#
from utils.config import Config
#========================================================#

class RobotFrame:
    def __init__(self):
        self.p_base_in_wf: np.ndarray = None
        self.r_base_to_wf: np.ndarray = None
        
        pass
    
    def updateFrame(self, config: pink.Configuration):
        self.update_pfFrame(config)
        pass
    
    @staticmethod
    def rotToEuler(r_link_to_frame):
        """ 回傳roll, pitch, yaw的順序(x,y,z), 以column vector"""
        return np.vstack((
            R.from_matrix(r_link_to_frame).as_euler('zyx', degrees=False)[::-1]
        ))
    
    @staticmethod
    def eularToGeometry(angle: np.ndarray):
        _, ay, az = angle.flatten()
        return np.array(
            [ cos(ay)*cos(az), -sin(az), 0 ],
            [ cos(ay)*sin(az),  cos(az), 0 ],
            [        -sin(ay),        0, 1 ],
        )
        
    def update_pfFrame(self, config: pink.Configuration):
        
        def _get_pfFrame(link: str):
            htm = config.get_transform_frame_to_world(link)
            p_in_pf = np.reshape( htm.translation, (3,1) )
            r_to_pf = np.reshape( htm.rotation, (3,3) )
        
            return p_in_pf, r_to_pf
        
        self.p_pel_in_pf,    self.r_pel_to_pf    = _get_pfFrame("pelvis_link")
        self.p_LhipX_in_pf,  self.r_LhipX_to_pf  = _get_pfFrame("l_hip_yaw_1")
        self.p_LhipZ_in_pf,  self.r_LhipZ_to_pf  = _get_pfFrame("l_hip_pitch_1")
        self.p_LhipY_in_pf,  self.r_LhipY_to_pf  = _get_pfFrame("l_thigh_1")
        self.p_LkneeY_in_pf, self.r_LkneeY_to_pf = _get_pfFrame("l_shank_1")
        self.p_LankY_in_pf,  self.r_LankY_to_pf  = _get_pfFrame("l_ankle_1")
        self.p_LankX_in_pf,  self.r_LankX_to_pf  = _get_pfFrame("l_foot_1")
        self.p_lf_in_pf,     self.r_lf_to_pf     = _get_pfFrame("l_foot")
        self.p_RhipX_in_pf,  self.r_RhipX_to_pf  = _get_pfFrame("r_hip_yaw_1")
        self.p_RhipZ_in_pf,  self.r_RhipZ_to_pf  = _get_pfFrame("r_hip_pitch_1")
        self.p_RhipY_in_pf,  self.r_RhipY_to_pf  = _get_pfFrame("r_thigh_1")
        self.p_RkneeY_in_pf, self.r_RkneeY_to_pf = _get_pfFrame("r_shank_1")
        self.p_RankY_in_pf,  self.r_RankY_to_pf  = _get_pfFrame("r_ankle_1")
        self.p_RankX_in_pf,  self.r_RankX_to_pf  = _get_pfFrame("r_foot_1")
        self.p_rf_in_pf,     self.r_rf_to_pf     = _get_pfFrame("r_foot")
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
        )

    def get_posture(self, p_pel_in_pf, p_lf_in_pf, p_rf_in_pf, r_pel_to_pf, r_lf_to_pf, r_rf_to_pf):
        
        a_pel_in_pf = self.rotToEuler(r_pel_to_pf)
        a_lf_in_pf  = self.rotToEuler(r_lf_to_pf) 
        a_rf_in_pf  = self.rotToEuler(r_rf_to_pf) 
        
        self.pa_pel_in_pf = np.vstack(( p_pel_in_pf, a_pel_in_pf ))
        self.pa_lf_in_pf  = np.vstack(( p_lf_in_pf,  a_lf_in_pf  ))
        self.pa_rf_in_pf  = np.vstack(( p_rf_in_pf,  a_rf_in_pf  ))
        
        L_Body_transfer = self.eularToGeometry(a_lf_in_pf)
        R_Body_transfer = self.eularToGeometry(a_rf_in_pf)

        pa_lfTOpel_in_pf = self.pa_pel_in_pf - self.pa_lf_in_pf #骨盆中心相對於左腳
        pa_rfTOpel_in_pf = self.pa_pel_in_pf - self.pa_rf_in_pf #骨盆中心相對於右腳

        return pa_lfTOpel_in_pf, pa_rfTOpel_in_pf, self.pa_pel_in_pf, self.pa_lf_in_pf, self.pa_rf_in_pf, L_Body_transfer, R_Body_transfer