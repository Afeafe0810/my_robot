#================ import library ========================#
from utils.signal_process import *
#========================================================#

class Config:
    TIMER_PERIOD = 0.01 #取樣頻率
    JNT_ORDER_SUB = (
        #訂閱到的關節順序，是錯的
        'L_Hip_Z', 'L_Hip_Y', 'L_Knee_Y', 'L_Ankle_Y', 
        'L_Ankle_X', 'R_Hip_X', 'R_Hip_Z', 'R_Knee_Y', 
        'R_Hip_Y', 'R_Ankle_Y', 'L_Hip_X', 'R_Ankle_X'
    )

    JNT_ORDER_LITERAL = [
        #實際上的關節順序
        'L_Hip_X', 'L_Hip_Z', 'L_Hip_Y', 'L_Knee_Y', 
        'L_Ankle_Y', 'L_Ankle_X', 'R_Hip_X', 'R_Hip_Z', 
        'R_Hip_Y', 'R_Knee_Y', 'R_Ankle_Y', 'R_Ankle_X'
    ]
    
    ROBOT_MODEL_DIR = "/home/ldsc/ros2_ws/src/bipedal_floating_description/urdf"
    DDT = 2
    
    #====================微分器====================#
    DIFFTER = {
        "p_com_in_wf" : Diffter(),
    }
    
    #====================濾波器====================#
    FILTER = {
        "v_com_in_wf" : Filter(),
    }
    
