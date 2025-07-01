from typing import Literal, TypeVar, ClassVar
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from bipedal_floating_description.utils.config import Config, Stance, GravityDict
from bipedal_floating_description.mode.state1 import AlipX as PreAlipX
from bipedal_floating_description.mode.utils import linear_move

NL = Config.NL_MOVINGTOLF
Hpel = Config.IDEAL_Z_PEL_IN_WF

End = dict[Literal['pel', 'lf', 'rf'], NDArray]
Ft = dict[Literal['lf', 'rf'], NDArray]

def gravity(model_gravity: GravityDict, end_in_pf: End) -> NDArray:
    end = end_in_pf
    # 根據骨盆位置來判斷重心腳
    y_ftTOpel = {
        'lf': abs(end['lf'][1] - end['pel'][1]),
        'rf': abs(end['rf'][1] - end['pel'][1])
    }
    
    gf = 'lf' if y_ftTOpel['lf'] <= y_ftTOpel['rf'] else 'rf'
    
    # 再根據重心腳距離權重
    return model_gravity[gf] * (1 - y_ftTOpel[gf]/0.1) + model_gravity['from_both_single_ft'] * (y_ftTOpel[gf]/0.1)

@dataclass
class State2:
    Tn: ClassVar[int] = 0
    is_just_started: ClassVar[bool] = True
    stance : ClassVar = Stance('lf', 'rf')
    
    end_in_wf: End
    p_end_in_pf: End
    a_end_in_pf: End
    _model_gravity: GravityDict
    jp: NDArray
    jv: NDArray
    w_from_Euler_to_geometry: Ft
    J: Ft
    
    def __post_init__(self):
        self.tauG = gravity(self._model_gravity, self.p_end_in_pf)

    def ctrl(self):
        cf, sf = self.stance
        if self.is_just_started:
            AlipX.from_previous_alip(PreAlipX.var_e, PreAlipX.bias_e)

        ref_p, ref_a, ref_varx, ref_ax_cf = Plan(
            self.Tn,
            self.is_just_started,
            self.end_in_wf
        ).plan()
        
        tau_knee = Knee(
            self.stance,
            {'lf': self.tauG[:6], 'rf': self.tauG[6:]},
            self.jv,
            (ref_p, ref_a),
            (self.p_end_in_pf, self.a_end_in_pf),
            self.w_from_Euler_to_geometry,
            self.J
        ).ctrl()
        
        
        sf_ankle = PD_Sf_Ankle(self.jp[-2:])
        
        cf_ankleX = PD_Cf_AnkleX(ref_ax_cf, self.jp[5], self.jv[5])
        cf_ankleY = AlipX(ref_varx, self.end_in_wf['pel'][0])
        
        tau_ankle = {
            cf: [cf_ankleY.ctrl(), cf_ankleX.ctrl()],
            sf: sf_ankle.ctrl()
        }
        
        self.__class__.is_just_started = False
        if self.Tn <= NL:
            self.__class__.Tn += 1
        
        return np.hstack((tau_knee['lf'], tau_ankle['lf'], tau_knee['rf'], tau_ankle['rf']))    

class Plan:
    pel0: NDArray
    lf0: NDArray
    rf0: NDArray
    
    def __init__(self, Tn: int, is_just_started: bool, end_in_wf: End):
        cls = self.__class__
        
        self.Tn = Tn
        if is_just_started:
            cls.pel0 = end_in_wf['pel']
            cls.lf0 = end_in_wf['lf']
            cls.rf0 = end_in_wf['rf']
        
    def plan(self) -> tuple[End, End, NDArray, float]:
        y_pel = linear_move(self.Tn, 0, NL, self.pel0[1], 0.09)
        
        ref_p_end: End = {
            'pel': np.array([self.pel0[0], y_pel, Hpel]),
            'lf': self.lf0,
            'rf': self.rf0
        }
        ref_a_end: End = {
            'pel': np.zeros(3),
            'lf': np.zeros(3),
            'rf': np.zeros(3)
        }
        ref_varx = np.array([0, 0])
        ref_ax_cf = linear_move(self.Tn, 0, NL, 0.02, 0.15)
        
        return ref_p_end, ref_a_end, ref_varx, ref_ax_cf

@dataclass
class Knee:
    stance: Stance
    tau_G: Ft
    _jv: NDArray
    ref: tuple[End, End]
    end_in_pf: tuple[End, End]
    w_from_Euler_to_geometry: Ft
    J: Ft
    
    def __post_init__(self):
        cf, sf = self.stance
        
        self.jv = {'lf': self._jv[:6], 'rf': self._jv[6:]}
        self.ctrltgt_idx = {
            cf: [2, 3, 4, 5], # z, ax, ay, az
            sf: [0, 1, 2, 5], # x, y, z, az
        }
        self.kout = {cf: np.array([25]), sf: np.array([20])}
        self.kin = {cf: np.array([1.5]), sf: np.array([0.5])}
        
    def ctrl(self)-> Ft:
        cf, sf = self.stance
        
        return {
            cf: self.ft_ctrl(cf),
            sf: self.ft_ctrl(sf)
        }
        
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

class PD_Sf_Ankle:
    ref_ayx_jp = np.zeros(2)
    kp = np.array([0.1])
    
    def __init__(self, jp45_sf: NDArray):
        self.jp45_sf = jp45_sf
        
    def ctrl(self) -> NDArray:
        """擺動腳腳踝的PD控制"""
        #TODO 摩擦力看要不要加 # HACK 現在只用P control
        return self.kp * ( self.ref_ayx_jp - self.jp45_sf ) 

class PD_Cf_AnkleX:
    kp: float = 4
    kd: float = 3
    
    def __init__(self, ref_jp: float, jp5_cf: float, jv5_cf: float):
        self.ref_jp = ref_jp
        self.jp5_cf = jp5_cf
        self.jv5_cf = jv5_cf
        
    def ctrl(self) -> float:
        return self.kp * (self.ref_jp - self.jp5_cf) - self.kd * self.jv5_cf

class AlipX:
    A = np.array([[1, 0.00247], [0.8832, 1]])
    B = np.array([0, 0.01])
    K = np.array([290.3274, 15.0198])*0.5
    L = np.array([1.656737, 24.448707])
    L_bias = -1.238861
    limit = Config.ANKLE_AY_LIMIT
    
    var_e: NDArray
    bias_e: float
    
    def __init__(self, ref_var: NDArray, pel_in_wf: float):
        self.ref_var = ref_var
        self.y = pel_in_wf
    
    def ctrl(self) -> float:
        #==========全狀態回授(飽和)==========#
        # _u = -self.K @ (np.vstack((self.var_e[0, 0] + self.bias_e, 0)) - ref_var)
        _u: float = -self.K @ (self.var_e - self.ref_var)
        u = np.clip(_u, -self.limit, self.limit)
        
        y_e = self.var_e[0] + self.bias_e
        err_e = self.y - y_e
        
        self.__class__.var_e = self.A @ self.var_e + self.B * u + self.L * err_e
        self.__class__.bias_e = self.bias_e + self.L_bias * err_e
        
        return -u
    
    @classmethod
    def from_previous_alip(cls, var_e: NDArray, bias_e: float):
        cls.var_e = var_e
        cls.bias_e = bias_e
        
if __name__ == '__main__':
    pass