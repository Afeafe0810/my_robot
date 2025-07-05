from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

# 資料結構體
GravityDict = dict[Literal['lf', 'rf', 'from_both_single_ft'], NDArray]
End = dict[Literal['lf', 'rf', 'pel'], NDArray]
Ft = dict[Literal['lf', 'rf'], NDArray]
FtScalar = dict[Literal['lf', 'rf'], float]

class Stance(NamedTuple):
    cf: Literal['lf', 'rf']
    sf: Literal['lf', 'rf']

# 參數與規格
class Config:
    '''機器人的參數、常數'''
    
    #取樣頻率
    Ts = 0.01
    
    # JointStates的最可能的順序, 順序無法改！！！！！
    # JNT_ORDER_SUB = (
    #     'L_Hip_Yaw', 'L_Hip_Pitch', 'L_Knee_Pitch', 'L_Ankle_Pitch', 'L_Ankle_Roll', 'R_Hip_Roll',
    #     'R_Hip_Yaw', 'R_Knee_Pitch', 'R_Hip_Pitch', 'R_Ankle_Pitch', 'L_Hip_Roll', 'R_Ankle_Roll' )

    #實際上URDF的關節順序以及effort controller的順序
    JNT_ORDER_LITERAL = (
        'L_Hip_Roll', 'L_Hip_Yaw', 'L_Hip_Pitch', 'L_Knee_Pitch', 'L_Ankle_Pitch', 'L_Ankle_Roll',
        'R_Hip_Roll', 'R_Hip_Yaw', 'R_Hip_Pitch', 'R_Knee_Pitch', 'R_Ankle_Pitch', 'R_Ankle_Roll'
    )
    #(X, Z, Y, Y, Y, X)
    
    #URDF檔的路徑
    ROBOT_MODEL_DIR = "/home/ldsc/ros2_ws/src/bipedal_floating_description/urdf"
    
    #骨盆相對base的座標
    P_PEL_IN_BASE = np.vstack(( 0, 0, 0.598 ))
    
    
    NL_BALANCE = 100 #雙腳平衡sample長
    NL_MOVINGTOLF = 100 #重心移動sample長
    NL_MARCHINPLACE = 50 #原地行走sample長
    
    #單腳支撐時間?
    DDT = 2
    
    #行走每步時間?
    STEP_TIMELENGTH = 0.5
    STEP_SAMPLELENGTH : int = int(STEP_TIMELENGTH / Ts)
    
    #機器人的物理模型
    MASS = 9
    IDEAL_Z_COM_IN_WF = 0.45
    IDEAL_Z_PEL_IN_WF = 0.55
    IDEAL_Y_STEPLENGTH = 0.2
    GC = 9.81
    OMEGA = ( GC / IDEAL_Z_COM_IN_WF )**0.5
    STEP_HEIGHT = 0.02
    
    #腳踝關節限制
    FOOT_WIDTH = 0.04
    FOOT_LENGTH = 0.142/2
    ANKLE_AX_LIMIT = MASS * GC * FOOT_WIDTH * 0.8 #3.53 Nm
    ANKLE_AY_LIMIT = MASS * GC * FOOT_LENGTH * 0.8 #3.53 Nm
    
    
    #state2 切到 state30 的初始角速度Lx
    INITIAL_LX = 0.245867
    ALIP_COLUMN_TITLE = [
        'com_x', 'com_y', 'com_z',
        'lf_x','lf_y','lf_z',
        'rf_x', 'rf_y', 'rf_z',
        'x','y',
        'Ly','Lx',
        'pel_x', 'pel_y', 'pel_z',
    ]