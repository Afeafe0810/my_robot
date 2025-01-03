import numpy as np
#================ import library ========================#

#========================================================#

class Config:
    '''機器人的參數、常數'''
    
    #取樣頻率
    TIMER_PERIOD = 0.01
    
    #訂閱到的關節順序，是錯的
    JNT_ORDER_SUB = (
        'L_Hip_Z'  , 'L_Hip_Y'  , 'L_Knee_Y', 'L_Ankle_Y', 
        'L_Ankle_X', 'R_Hip_X'  , 'R_Hip_Z' , 'R_Knee_Y' , 
        'R_Hip_Y'  , 'R_Ankle_Y', 'L_Hip_X' , 'R_Ankle_X',
    )
    
    #實際上的關節順序
    JNT_ORDER_LITERAL = (
        'L_Hip_X'  , 'L_Hip_Z'  , 'L_Hip_Y'  , 'L_Knee_Y' , 
        'L_Ankle_Y', 'L_Ankle_X', 'R_Hip_X'  , 'R_Hip_Z'  , 
        'R_Hip_Y'  , 'R_Knee_Y' , 'R_Ankle_Y', 'R_Ankle_X',
    )
    
    #URDF檔的路徑
    ROBOT_MODEL_DIR = "/home/ldsc/ros2_ws/src/bipedal_floating_description/urdf"
    
    #骨盆相對base的座標
    P_PEL_IN_BASE = np.vstack(( 0, 0, 0.598 ))
    
    #單腳支撐時間?
    DDT = 2
    
    
    
    
    
