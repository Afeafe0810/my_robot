#================ import library ========================#
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray 




import pinocchio as pin

import pink


import numpy as np
np.set_printoptions(precision=2)

from sys import argv
from os.path import dirname, join, abspath
import copy
import math
from scipy.spatial.transform import Rotation as R

import csv
#================ import other code =====================#
from utils.robot_control_framesensor import ULC_frame
#========================================================#

class ULC_traj:
    
    @staticmethod
    def ref_cmd(ulc, state,px_in_lf,px_in_rf,stance,com_in_lf,com_in_rf):
        ulc.PX_ref = np.array([[0.0],[0.0],[0.57],[0.0],[0.0],[0.0]])
        ulc.LX_ref = np.array([[0.0],[0.1],[0.0],[0.0],[0.0],[0.0]])
        ulc.RX_ref = np.array([[0.0],[-0.1],[0.0],[0.0],[0.0],[0.0]])
        
        # if state in [0, 30]:
        #     self.PX_ref = np.array([[0.0],[0.0],[0.57],[0.0],[0.0],[0.0]])
        #     self.LX_ref = np.array([[0.0],[0.1],[0.0],[0.0],[0.0],[0.0]])
        #     self.RX_ref = np.array([[0.0],[-0.1],[0.0],[0.0],[0.0],[0.0]])
        # else:
        #     # Lth = 0.16
        #     # hLth = 0.06
        #     # hhLth = 0.03
        #     # pyLth = 0.06
        #     # hight  = 0.03
        #     hLth = 0.0
        #     hhLth = 0.0
        #     pyLth = -0.06
        #     hight  = 0.0
        #     if state == 1:
        #         R_X_ref = 0.0
        #         R_Z_ref = 0.0
        #         if self.DS_time > 0.0 and self.DS_time <= self.DDT:
        #             if self.DS_time > (0.5*self.DDT) and self.DS_time <= self.DDT:
        #                 P_X_ref = 0.0
        #                 P_Y_ref = -pyLth
        #                 if self.DS_time <= (0.75*self.DDT):
        #                     L_X_ref = -hhLth*((self.DS_time-(0.5*self.DDT))/(0.25*self.DDT))
        #                     L_Z_ref = hight*((self.DS_time-(0.5*self.DDT))/(0.25*self.DDT))
        #                 else:
        #                     L_X_ref = -hhLth-hhLth*((self.DS_time-(0.75*self.DDT))/(0.25*self.DDT))
        #                     L_Z_ref = hight-hight*((self.DS_time-(0.75*self.DDT))/(0.25*self.DDT))                            
        #             else:
        #                 P_X_ref = 0.0
        #                 P_Y_ref = -pyLth*(self.DS_time/(0.5*self.DDT))
        #                 L_X_ref = 0.0
        #                 L_Z_ref = 0.0
        #         else:
        #             P_X_ref = 0.0
        #             P_Y_ref = -pyLth
        #             L_X_ref = -hLth
        #             L_Z_ref = 0.0

        #     if state == 2:
        #         if stance == 2:
        #             L_Z_ref = 0.0
        #             R_Z_ref = 0.0
        #             if self.DS_time > 0.0 and self.DS_time <= self.DDT:
        #                 if self.RSS_count == 0:
        #                     P_X_ref = hhLth+hhLth*(self.DS_time/self.DDT)
        #                     P_Y_ref = -pyLth*(self.DS_time/self.DDT)
        #                     L_X_ref = 0.0
        #                     R_X_ref = hLth
        #                 else:
        #                     P_X_ref = hhLth+hhLth*(self.DS_time/self.DDT)
        #                     P_Y_ref = pyLth*(self.DS_time/self.DDT)
        #                     L_X_ref = hLth
        #                     R_X_ref = 0.0
        #             else:
        #                 if abs(px_in_rf[1,0])<=0.08:
        #                     P_X_ref = hLth
        #                     P_Y_ref = -pyLth
        #                     L_X_ref = 0.0
        #                     R_X_ref = hLth
        #                 elif abs(px_in_lf[1,0])<=0.08:
        #                     P_X_ref = hLth
        #                     P_Y_ref = pyLth
        #                     L_X_ref = hLth
        #                     R_X_ref = 0.0
        #                 else:
        #                     P_Y_ref = 0.0

        #         elif stance == 0:
        #             R_X_ref = 0.0
        #             R_Z_ref = 0.0
        #             fq_RDT = 0.25*self.RDT
        #             h_RDT = 0.5*self.RDT
        #             rq_RDT = 0.75*self.RDT
        #             if self.RSS_time > 0.0 and self.RSS_time<=self.RDT:
        #                 if self.RSS_time > fq_RDT and self.RSS_time <= h_RDT:
        #                     P_X_ref = 0.0
        #                     P_Y_ref = -pyLth
        #                     L_X_ref = -hLth+hLth*((self.RSS_time-fq_RDT)/(h_RDT-fq_RDT)) #lift l leg
        #                     L_Z_ref = hight*((self.RSS_time-fq_RDT)/(h_RDT-fq_RDT)) #lift l leg
        #                 elif self.RSS_time > h_RDT and self.RSS_time <= rq_RDT:
        #                     P_X_ref = 0.0
        #                     P_Y_ref = -pyLth
        #                     # P_X_ref = hhLth*((self.RSS_time-h_RDT)/(rq_RDT-h_RDT)) #lay down l leg
        #                     # P_Y_ref = -pyLth+pyLth*((self.RSS_time-h_RDT)/(rq_RDT-h_RDT)) #lay down l leg
        #                     L_X_ref = hLth*((self.RSS_time-h_RDT)/(rq_RDT-h_RDT)) #lay down l leg
        #                     L_Z_ref = hight-hight*((self.RSS_time-h_RDT)/(rq_RDT-h_RDT)) #lay down l leg
        #                 elif self.RSS_time > rq_RDT:
        #                     P_X_ref = hhLth*((self.RSS_time-rq_RDT)/(self.RDT-rq_RDT)) #lay down l leg
        #                     P_Y_ref = -pyLth+pyLth*((self.RSS_time-rq_RDT)/(self.RDT-rq_RDT)) #lay down l leg
        #                     # P_X_ref = hhLth
        #                     # P_Y_ref = 0.0
        #                     L_X_ref = hLth
        #                     L_Z_ref = 0.0
        #                 else:
        #                     P_X_ref = 0.0
        #                     P_Y_ref = -pyLth
        #                     L_X_ref = -hLth
        #                     L_Z_ref = 0.0
        #             else:
        #                 P_X_ref = hhLth
        #                 P_Y_ref = 0.0
        #                 L_X_ref = hLth
        #                 L_Z_ref = 0.0
                
        #         else: #stance = 1
        #             L_X_ref = 0.0
        #             L_Z_ref = 0.0
        #             fq_LDT = 0.25*self.LDT
        #             h_LDT = 0.5*self.LDT
        #             rq_LDT = 0.75*self.LDT
        #             if self.LSS_time > 0.0 and self.LSS_time <= self.LDT:
        #                 if self.LSS_time > fq_LDT and self.LSS_time <= h_LDT:
        #                     P_X_ref = 0.0
        #                     P_Y_ref = pyLth
        #                     R_X_ref = -hLth+hLth*((self.LSS_time-fq_LDT)/(h_LDT-fq_LDT)) #lift r leg
        #                     R_Z_ref = hight*((self.LSS_time-fq_LDT)/(h_LDT-fq_LDT)) #lift r leg
        #                 elif self.LSS_time > h_LDT and self.LSS_time <= rq_LDT:
        #                     P_X_ref = 0.0
        #                     P_Y_ref = pyLth
        #                     # P_X_ref = hhLth*((self.LSS_time-h_LDT)/(rq_LDT-h_LDT)) #lay down r leg
        #                     # P_Y_ref = pyLth-pyLth*((self.LSS_time-h_LDT)/(rq_LDT-h_LDT)) #lay down r leg
        #                     R_X_ref = hLth*((self.LSS_time-h_LDT)/(rq_LDT-h_LDT)) #lay down r leg
        #                     R_Z_ref = hight-hight*((self.LSS_time-h_LDT)/(rq_LDT-h_LDT)) #lay down r leg
        #                 elif self.LSS_time > rq_LDT:
        #                     P_X_ref = hhLth*((self.LSS_time-rq_LDT)/(self.LDT-rq_LDT)) #lay down r leg
        #                     P_Y_ref = pyLth-pyLth*((self.LSS_time-rq_LDT)/(self.LDT-rq_LDT)) #lay down r leg
        #                     # P_X_ref = hhLth
        #                     # P_Y_ref = 0.0
        #                     R_X_ref = hLth
        #                     R_Z_ref = 0.0
        #                 else:
        #                     P_X_ref = 0.0
        #                     P_Y_ref = pyLth #
        #                     R_X_ref = -hLth
        #                     R_Z_ref = 0.0
        #             else:
        #                 P_X_ref = hhLth
        #                 P_Y_ref = 0.0
        #                 R_X_ref = hLth
        #                 R_Z_ref = 0.0
            
        #     #pelvis
        #     P_Z_ref = 0.55
        #     P_Roll_ref = 0.0
        #     P_Pitch_ref = 0.0
        #     P_Yaw_ref = 0.0

        #     #left_foot
        #     L_Y_ref = 0.1
        #     L_Roll_ref = 0.0
        #     L_Pitch_ref = 0.0
        #     L_Yaw_ref = 0.0
            
        #     #right_foot
        #     R_Y_ref = -0.1           
        #     R_Roll_ref = 0.0
        #     R_Pitch_ref = 0.0
        #     R_Yaw_ref = 0.0

        #     self.PX_ref = np.array([[P_X_ref],[P_Y_ref],[P_Z_ref],[P_Roll_ref],[P_Pitch_ref],[P_Yaw_ref]])
        #     self.LX_ref = np.array([[L_X_ref],[L_Y_ref],[L_Z_ref],[L_Roll_ref],[L_Pitch_ref],[L_Yaw_ref]])
        #     self.RX_ref = np.array([[R_X_ref],[R_Y_ref],[R_Z_ref],[R_Roll_ref],[R_Pitch_ref],[R_Yaw_ref]])  
        
        # return
