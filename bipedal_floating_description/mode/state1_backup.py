import numpy as np; np.set_printoptions(precision=2)
from numpy.typing import NDArray
from typing import Literal
#================ import library ========================#
from bipedal_floating_description.utils.robot_model import RobotModel
from bipedal_floating_description.utils.frame_kinermatic import RobotFrame
from bipedal_floating_description.mode.utils import linear_move, Stance
from bipedal_floating_description.utils.config import Config, GravityDict
import bipedal_floating_description.torque_control.pd_control as PD
from bipedal_floating_description.torque_control.alip_control import AlipX
#========================================================#

FtDict = dict[Literal['lf', 'rf'], NDArray]
EndInPf = dict[Literal['lf', 'rf', 'pel'], NDArray]


Hpel = Config.IDEAL_Z_PEL_IN_WF
NL = Config.TL_BALANCE

alipx = AlipX()

def gravity(model_gravity: GravityDict, end_in_pf: EndInPf) -> NDArray:
    end = end_in_pf
    # 根據骨盆位置來判斷重心腳
    y_ftTOpel = {
        'lf': abs(end['lf'][1] - end['pel'][1]),
        'rf': abs(end['rf'][1] - end['pel'][1])
    }
    
    gf = 'lf' if y_ftTOpel['lf'] <= y_ftTOpel['rf'] else 'rf'
    
    # 再根據重心腳距離權重
    return model_gravity[gf] * (1 - y_ftTOpel[gf]/0.1) + model_gravity['from_both_single_ft'] * (y_ftTOpel[gf]/0.1)
    

class State1:
    stance = Stance(cf='lf', sf='rf')
    kout = {stance.cf: 25, stance.sf: 20}
    kin = {stance.cf: 0.5, stance.sf: 0.5}
    Tn: int = 0
    is_just_started: bool = True
    pel0: NDArray
    lf0: NDArray
    rf0: NDArray
    
    def __init__(
        self,
        end_in_pf: dict[str, NDArray],
        end_in_wf: dict[str, NDArray],
        w_from_Euler_to_geometry: dict[str, NDArray],
        jp: NDArray,
        jv: NDArray,
        Jacobian: dict[str, NDArray],
        model_gravity: dict[str, NDArray],
        frame: RobotFrame
    ):
        # pf端末位置, 用於內外環計算
        self.end_in_pf = end_in_pf
        self.end_in_wf = end_in_wf
        
        # wf端末位置, 用於規劃軌跡
        cls = self.__class__
        if cls.is_just_started:
            cls.pel0 = end_in_wf['pel']
            cls.lf0 = end_in_wf['lf']
            cls.rf0 = end_in_wf['rf']
        
        # 用於位置環將歐拉角的微分轉成幾何的角速度
        self.w_from_Euler_to_geometry = w_from_Euler_to_geometry
        self.jp = jp
        self.jv = jv
        self.J = Jacobian
        self.tau_G = gravity(model_gravity, end_in_pf)
        self.frame = frame
        
        # 規劃的軌跡點
        self.ref = self.plan()
        
        # 擺動腳踝pd控制的ref
        self.ref_jp = np.zeros(2)
        
    def plan(self):
        z_pel = linear_move(self.Tn, 0, NL, self.pel0[2], Hpel)
        return {
            'pel': np.hstack((self.pel0[:2], z_pel)),
            'lf': self.lf0,
            'rf': self.rf0,
            'a_pel': np.zeros(3),
            'a_lf': np.zeros(3),
            'a_rf': np.zeros(3)
        }
    
    def knee(self) -> NDArray:
        cf, sf = self.stance
        end = self.end_in_pf
        ref = self.ref
        
        # 控制相對骨盆的向量, 計算回授誤差
        err_p_pelTOcf = (ref[cf] - ref['pel']) - (end[cf] - end['pel'])
        err_p_pelTOsf = (ref[sf] - ref['pel']) - (end[sf] - end['pel'])
        err_a_pelTOcf = (ref[f'a_{cf}'] - ref['a_pel']) - (end[f'a_{cf}'] - end['a_pel'])
        err_a_pelTOsf = (ref[f'a_{sf}'] - ref['a_pel']) - (end[f'a_{sf}'] - end['a_pel'])
        
        # 外環 P gain 作為微分
        v_pelTOcf = self.kout[cf] * err_p_pelTOcf
        v_pelTOsf = self.kout[sf] * err_p_pelTOsf
        da_pelTOcf = self.kout[cf] * err_a_pelTOcf
        da_pelTOsf = self.kout[sf] * err_a_pelTOsf
        
        # 「歐拉角的微分」轉成「幾何角速度」
        w_pelTOcf = self.w_from_Euler_to_geometry[cf] @ da_pelTOcf
        w_pelTOsf = self.w_from_Euler_to_geometry[sf] @ da_pelTOsf
        
        """這邊在處理微分運動學，將端末速度轉成關節速度命令
        
        控制目標:
            支撐腳膝上4關節控制骨盆的z, ax, ay, az; 擺動腳膝上4關節控制擺動腳掌x, y, z, az
        微分運動學的關係:
            v_target = J_kneeTOtarget @ dq_knee + J_ankleTOtarget @ dq_ankle
        故只想控膝蓋以上，需要扣掉腳踝的影響
            dq_knee = J_kneeTOtarget^-1 @ (v_target - J_ankleTOtarget @ dq_ankle)
        """
        
        knee_idx = [0, 1, 2, 3]
        ankle_idx = [4, 5]
        target_idx = {
            cf: [2, 3, 4, 5], # z, ax, ay, az
            sf: [0, 1, 2, 5], # x, y, z, az
        }
        
        dq_ankle = {
            cf: self.jv[:6][ankle_idx],
            sf: self.jv[6:][ankle_idx]
        }
        target = {
            cf: np.hstack([v_pelTOcf[2], w_pelTOcf]),
            sf: np.hstack([v_pelTOsf, w_pelTOsf[2]])
        }
        
        J_kneeTOtarget = {
            cf: self.J[cf][np.ix_(target_idx[cf], knee_idx)],
            sf: self.J[sf][np.ix_(target_idx[sf], knee_idx)]
        }
        
        J_ankleTOtarget = {
            cf: self.J[cf][np.ix_(target_idx[cf], ankle_idx)],
            sf: self.J[sf][np.ix_(target_idx[sf], ankle_idx)]
        }
        
        cmd_dq_knee = {
            cf: np.linalg.pinv(J_kneeTOtarget[cf]) @ (target[cf] - J_ankleTOtarget[cf] @ dq_ankle[cf]),
            sf: np.linalg.pinv(J_kneeTOtarget[sf]) @ (target[sf] - J_ankleTOtarget[sf] @ dq_ankle[sf])
        }
        
        # 進入內環動力學
        dq_knee = {
            cf: self.jv[:6][knee_idx],
            sf: self.jv[6:][knee_idx]
        }
        tau_G_knee = {
            cf: self.tau_G[:6][knee_idx],
            sf: self.tau_G[6:][knee_idx]
        }
        print('1', cmd_dq_knee[cf])
        print('2', dq_knee[cf])
        return {
            cf: self.kin[cf] * (cmd_dq_knee[cf] - dq_knee[cf]) + tau_G_knee[cf],
            sf: self.kin[sf] * (cmd_dq_knee[sf] - dq_knee[sf]) + tau_G_knee[sf]
        }
    
    def pd_sfAnkle(self):
        axyz_sf_in_wf = self.end_in_wf[f'a_{self.stance.sf}']
        ayx_sf_in_wf = axyz_sf_in_wf[:1:-1]
        #TODO 摩擦力看要不要加
        return 0.1 * (self.ref_jp - ayx_sf_in_wf ) # HACK 現在只用P control
    
    def ctrl(self):
        tau_knee = self.knee()
        tau_ankle_Cf_ay = alipx.ctrl(self.frame, self.stance, self.stance, None, None).flatten()
        tau_ankle_Cf_ax = PD.ankle_ax1_cf(None, None, self.jp, self.jv)
        print(f"{tau_ankle_Cf_ay = }\n{tau_ankle_Cf_ax = }")
        tau_ankle = {
            self.stance.cf: np.hstack((tau_ankle_Cf_ay, tau_ankle_Cf_ax)),
            self.stance.sf: self.pd_sfAnkle().flatten()
        }
        
        cls = self.__class__
        cls.Tn += 1
        cls.is_just_started = False
        print(tau_knee['lf'], tau_ankle['lf'],
            tau_knee['rf'], tau_ankle['rf'])
        return np.hstack((
            tau_knee['lf'], tau_ankle['lf'],
            tau_knee['rf'], tau_ankle['rf']
        ))