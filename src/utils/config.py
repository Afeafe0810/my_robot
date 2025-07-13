from typing import Literal, NamedTuple, TypeAlias, Final
import os

import numpy as np
from numpy.typing import NDArray


""" All pure stucture Used """
# 向量與矩陣
Arr: TypeAlias =  NDArray[np.floating]
Vec: TypeAlias =  NDArray[np.floating]
Mat: TypeAlias =  NDArray[np.floating]

# robot_model重力矩回傳的結構
GravityDict: TypeAlias = dict[Literal['lf', 'rf', 'from_both_single_ft'], Arr]

End = dict[Literal['lf', 'rf', 'pel'], Arr]

Ft = dict[Literal['lf', 'rf'], Arr]

FtScalar = dict[Literal['lf', 'rf'], float|np.floating]

class Stance(NamedTuple):
    cf: Literal['lf', 'rf']
    sf: Literal['lf', 'rf']


def _get_ros2pkg_abspath() -> str:
    """ 回傳當前電腦的ros2pkg的絕對路徑 """
    pkg_name = 'bipedal_floating_description'
    dir_buildedpkg = os.path.abspath(__file__).rsplit(pkg_name, 1)[0] + pkg_name
    dir_ros2ws = os.path.dirname(os.path.dirname(dir_buildedpkg))
    dir_pkg = os.path.join(dir_ros2ws, 'src', pkg_name)
    return dir_pkg


class Config:
    """ 腳本用到的參數, 規格, 該注意的順序 """
    
    
    """ 路徑們 """
    _dir_ros2pkg = _get_ros2pkg_abspath()
    DIR_URDF: Final[str] = os.path.join(_dir_ros2pkg, 'urdf')
    DIR_OUTPUT: Final[str] = os.path.join(_dir_ros2pkg, 'output')
    
    
    """ 取樣時間與不同模式設定的時間 """
    Ts: Final[float] = 0.01
    NL_BALANCE: Final[int] = 100 #雙腳平衡sample長
    NL_MOVINGTOLF: Final[int] = 100 #重心移動sample長
    NL_MARCHINPLACE: Final[int] = 25 #原地行走sample長
    
    """ JointStates回傳時最可能的順序, 順序如何打亂由ROS決定, 無法改!!!! """
    # JNT_ORDER_SUB = (
    #     'L_Hip_Yaw', 'L_Hip_Pitch', 'L_Knee_Pitch', 'L_Ankle_Pitch', 'L_Ankle_Roll', 'R_Hip_Roll',
    #     'R_Hip_Yaw', 'R_Knee_Pitch', 'R_Hip_Pitch', 'R_Ankle_Pitch', 'L_Hip_Roll', 'R_Ankle_Roll' )


    """ 實際上URDF的關節順序以及effort controller的順序 """
    #(X, Z, Y, Y, Y, X)
    JNT_ORDER_LITERAL: Final[tuple[str, ...]] = (
        'L_Hip_Roll', 'L_Hip_Yaw', 'L_Hip_Pitch', 'L_Knee_Pitch', 'L_Ankle_Pitch', 'L_Ankle_Roll',
        'R_Hip_Roll', 'R_Hip_Yaw', 'R_Hip_Pitch', 'R_Knee_Pitch', 'R_Ankle_Pitch', 'R_Ankle_Roll'
    )
    
    
    
    #骨盆相對base的座標
    # P_PEL_IN_BASE: Final[Arr] = np.vstack(( 0, 0, 0.598 ))
    
    
    """ 機器人的物理參數 """
    MASS: Final[float] = 9
    IDEAL_Z_COM_IN_WF: Final[float] = 0.45
    IDEAL_Z_PEL_IN_WF: Final[float] = 0.55
    IDEAL_Y_STEPLENGTH: Final[float] = 0.2
    GC: Final[float] = 9.81
    OMEGA: Final[float] = ( GC / IDEAL_Z_COM_IN_WF )**0.5
    STEP_HEIGHT: Final[float] = 0.02
    
    #腳踝關節限制
    FOOT_WIDTH = 0.04
    FOOT_LENGTH = 0.142/2
    ANKLE_AX_LIMIT = MASS * GC * FOOT_WIDTH * 0.8 # 2.83 Nm
    ANKLE_AY_LIMIT = MASS * GC * FOOT_LENGTH * 0.8 # 5 Nm