from typing import Literal, TypeVar, ClassVar
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from src.utils.config import Config, Stance, GravityDict, End, Ft
from src.mode.utils import linear_move

NL = Config.NL_BALANCE
Hpel = Config.IDEAL_Z_PEL_IN_WF

def gravity(model_gravity: GravityDict, end_in_pf: End) -> Ft:
    end = end_in_pf
    # 根據骨盆位置來判斷重心腳
    y_ftTOpel = {
        'lf': abs(end['lf'][1] - end['pel'][1]),
        'rf': abs(end['rf'][1] - end['pel'][1])
    }
    
    gf = 'lf' if y_ftTOpel['lf'] <= y_ftTOpel['rf'] else 'rf'
    
    # 再根據重心腳距離權重
    tau = model_gravity[gf] * (1 - y_ftTOpel[gf]/0.1) + model_gravity['from_both_single_ft'] * (y_ftTOpel[gf]/0.1)
    return {'lf': tau[:6], 'rf': tau[6:]}


class PD_Sf_Ankle:
    ref_ayx_jp = np.zeros(2)
    kp = np.array([0.1])
    
    def ctrl(self, jp45_sf: NDArray) -> NDArray:
        """擺動腳腳踝的PD控制"""
        #TODO 摩擦力看要不要加 # HACK 現在只用P control
        return self.kp * ( self.ref_ayx_jp - jp45_sf ) 


class PD_Cf_AnkleX:
    ref_jp: float = 0.02
    kp: float = 4
    kd: float = 3
        
    def ctrl(self, jp5_cf: float, jv5_cf: float) -> float:
        return self.kp * (self.ref_jp - jp5_cf) - self.kd * jv5_cf


class AlipX:
    A = np.array([[1, 0.00247], [0.8832, 1]])
    B = np.array([0, 0.01])
    K = np.array([150,15.0198])*0.13
    L = np.array([1.656737, 24.448707])
    L_bias = -1.238861
    limit = Config.ANKLE_AY_LIMIT
    
    def __init__(self):
        self.var_e: NDArray = np.zeros(2)
        self.bias_e: float = -0.02
    
    def ctrl(self, ref_var: NDArray, cf2pel_in_wf: float) -> float:
        y = cf2pel_in_wf
        
        #==========全狀態回授(飽和)==========#
        _u: float = (-self.K @ (self.var_e - ref_var)).item()
        u = np.clip(_u, -self.limit, self.limit)
        
        y_e = self.var_e[0] + self.bias_e
        err_e = y - y_e
        
        self.var_e = self.A @ self.var_e + self.B * u + self.L * err_e
        self.bias_e = self.bias_e + self.L_bias * err_e
        
        return -u


@dataclass
class Knee:
    stance: Stance
    tau_G: Ft
    jv: Ft
    ref: tuple[End, End]
    end_in_pf: tuple[End, End]
    w_from_Euler_to_geometry: Ft
    J: Ft
    
    
    def __post_init__(self):
        cf, sf = self.stance
        
        self.ctrltgt_idx = {
            cf: [2, 3, 4, 5], # z, ax, ay, az
            sf: [0, 1, 2, 5], # x, y, z, az
        }
        self.kout = {cf: np.array([25]), sf: np.array([20])}
        self.kin = {cf: np.array([0.5]), sf: np.array([0.5])}
        
  
    def ft_ctrl(self, ft: Literal['lf', 'rf']) -> NDArray:
        """內外環的控制程式

        外環需計算誤差後須將歐拉角微分轉成幾何的角動量
        
        微分運動學: 將端末速度轉成關節速度命令
            - 控制目標:
                支撐腳膝上4關節控制骨盆的z, ax, ay, az; 擺動腳膝上4關節控制擺動腳掌x, y, z, az
            - 微分運動學的關係:
                v_target = J_kneeTOtarget @ dq_knee + J_ankleTOtarget @ dq_ankle
            - 故只想控膝蓋以上，需要扣掉腳踝的影響
                dq_knee = J_kneeTOtarget^-1 @ (v_target - J_ankleTOtarget @ dq_ankle)
        """
        p_end, a_end = self.end_in_pf
        p_ref, a_ref = self.ref
        
        # 控制相對骨盆的向量, 計算回授誤差
        err_p_pelTOft = (p_ref[ft] - p_ref['pel']) - (p_end[ft] - p_end['pel'])
        err_a_pelTOft = (a_ref[ft] - a_ref['pel']) - (a_end[ft] - a_end['pel'])
        
        # 外環 P gain 作為微分
        v_pelTOft = self.kout[ft] * err_p_pelTOft
        da_pelTOft = self.kout[ft] * err_a_pelTOft
        
        # 「歐拉角的微分」轉成「幾何角速度」
        w_pelTOft = self.w_from_Euler_to_geometry[ft] @ da_pelTOft
        
        # 微分運動學
        knee_idx = [0, 1, 2, 3]
        ankle_idx = [4, 5]
        
        dq_ankle = self.jv[ft][ankle_idx]
        ctrltgt = np.hstack([v_pelTOft, w_pelTOft])[self.ctrltgt_idx[ft]]
        
        J_kneeTOtarget = self.J[ft][np.ix_(self.ctrltgt_idx[ft], knee_idx)]
        J_ankleTOtarget = self.J[ft][np.ix_(self.ctrltgt_idx[ft], ankle_idx)]
        
        cmd_dq_knee = np.linalg.pinv(J_kneeTOtarget) @ (ctrltgt - J_ankleTOtarget @ dq_ankle)
        
        # 進入內環動力學
        dq_knee = self.jv[ft][knee_idx]
        tau_G_knee = self.tau_G[ft][knee_idx]
        return self.kin[ft] * (cmd_dq_knee - dq_knee) + tau_G_knee
    
    
    def ctrl(self)-> Ft:       
        return {
            'lf': self.ft_ctrl('lf'),
            'rf': self.ft_ctrl('rf')
        }
      

class Plan:
    
    def __init__(self, end_in_wf: End):
        self.pel0: NDArray = end_in_wf['pel']
        self.lf0: NDArray = end_in_wf['lf']
        self.rf0: NDArray = end_in_wf['rf']
    
    
    def plan(self, Tn: int)-> tuple[End, End, NDArray]:
        
        z_pel = linear_move(Tn, 0, NL, self.pel0[2], Hpel)
        
        ref_p_end: End = {
            'pel': np.hstack((self.pel0[:2], z_pel)),
            'lf': self.lf0,
            'rf': self.rf0
        }
        ref_a_end: End = {
            'pel': np.zeros(3),
            'lf': np.zeros(3),
            'rf': np.zeros(3)
        }
        ref_varx = np.array([0, 0])
        
        return ref_p_end, ref_a_end, ref_varx


class State1:
    
    def __init__(self):
        self.Tn: int = 0
        self.has_run: bool = False
        self.stance: Stance = Stance('lf', 'rf')
        
        self.alipx: AlipX = AlipX()
        self.plan: Plan
    
    def ctrl(
        self,
        end_in_wf: End,
        p_end_in_pf: End,
        a_end_in_pf: End,
        _model_gravity: GravityDict,
        jp: Ft,
        jv: Ft,
        w_from_Euler_to_geometry: Ft,
        J: Ft
        
    )-> NDArray:
        
        cf, sf = self.stance
        
        """ 重力矩 """
        tauG = gravity(_model_gravity, p_end_in_pf)
        
        if not self.has_run:
            self.has_run = True
            self.plan = Plan(end_in_wf)
        
        """ 軌跡規劃 """
        ref_p, ref_a, ref_varx = self.plan.plan(self.Tn)
        
        """ 膝上關節扭矩 """
        tau_knee = Knee(
            self.stance,
            tauG,
            jv,
            (ref_p, ref_a),
            (p_end_in_pf, a_end_in_pf),
            w_from_Euler_to_geometry,
            J
        ).ctrl()
        
        """ 腳踝關節扭矩 """
        x_cf2pel = end_in_wf['pel'][0] - end_in_wf[cf][0]

        tau_ankle = {
            cf: [
                self.alipx.ctrl(ref_varx, x_cf2pel),
                PD_Cf_AnkleX().ctrl(jp[cf][5], jv[cf][5])
            ],
            sf: PD_Sf_Ankle().ctrl(jp[sf][4:6])
        }
        
        """ 更新時間 """
        self.Tn += 1
        
        return np.hstack([tau_knee['lf'], tau_ankle['lf'], tau_knee['rf'], tau_ankle['rf']])
