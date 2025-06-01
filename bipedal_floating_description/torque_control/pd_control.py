import numpy as np; np.set_printoptions(precision=5)
from numpy.typing import NDArray
from typing import TypeVar
#================ import library ========================#
from bipedal_floating_description.utils.robot_model import RobotModel
from bipedal_floating_description.utils.frame_kinermatic import RobotFrame
from bipedal_floating_description.utils.config import Config
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

def ankle_ax_sf(frame: RobotFrame, sf: str) -> NDArray:
    """支撐腳腳踝的PD控制"""
    ref_jp = np.zeros((2,1))

    r_ft_to_wf = {
        'lf': frame.r_lf_to_wf,
        'rf': frame.r_rf_to_wf
    }
    ayx_sf_in_wf = frame.rotMat_to_euler(r_ft_to_wf[sf]) [1::-1]
    #TODO 摩擦力看要不要加
    torque_ankle_sf = 0.1 * ( ref_jp - ayx_sf_in_wf ) # HACK 現在只用P control
    
    return torque_ankle_sf