import numpy as np; np.set_printoptions(precision=5)
from numpy.typing import NDArray
from typing import TypeVar
#================ import library ========================#
from utils.robot_model import RobotModel
from utils.frame_kinermatic import RobotFrame
from utils.config import Config
#========================================================#
FA = TypeVar('FloatOrArray', float, NDArray[np.float_])

def _control_jp(ref_jp: FA, jp: FA, jv: FA, kp: FA, kd: FA, tauG: FA = None) -> FA:
    if not np.shape(ref_jp) == np.shape(jp) == np.shape(jv) == np.shape(kp) == np.shape(kd):
        raise ValueError('!! size is not the same !!')
    
    tau = kp * (ref_jp - jp) - kd * jv
    
    if tauG is None:
        return tau
    else:
        return tau + tauG

def initial_balance(frame: RobotFrame, robot: RobotModel, jp: NDArray, jv: NDArray) -> NDArray:
    """在剛開機狀態直接用關節角度的 單環 來平衡"""
    return _control_jp(
        ref_jp = np.zeros((12,1)),
        jp = jp,
        jv = jv,
        kp = np.vstack([ 2, 2, 4, 6, 6, 4 ]*2),
        kd = np.zeros((12,1)),
        tauG = robot.gravity(jp, 0, ['lf', 'rf'], *frame.get_posture())
    )

def ankle_ax1_cf(frame: RobotFrame, robot: RobotModel, jp: NDArray, jv: NDArray) -> float:
    """state1 雙支撐左右方向的平衡"""
    _tau =  _control_jp(
        ref_jp = 0.02,
        jp = jp[5, 0],
        jv = jv[5, 0],
        kp = 4,
        kd = 3,
        # tauG = robot.gravity(jp, 0, ['lf', 'rf'], *frame.get_posture())[5, 0]
    )
    
    return np.clip(_tau, -Config.ANKLE_AX_LIMIT, Config.ANKLE_AX_LIMIT)

def ankle_ax2_cf(frame: RobotFrame, robot: RobotModel, jp: NDArray, jv: NDArray, ref_ax: float) -> float:
    """state2 重心移到左腳左右方向的平衡"""
    _tau =  _control_jp(
        ref_jp = ref_ax,
        jp = jp[5, 0],
        jv = jv[5, 0],
        kp = 4,
        kd = 3,
        # tauG = robot.gravity(jp, 0, ['lf', 'rf'], *frame.get_posture())[5, 0]
    )
    
    return np.clip(_tau, -Config.ANKLE_AX_LIMIT, Config.ANKLE_AX_LIMIT)
