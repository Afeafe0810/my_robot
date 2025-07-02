from typing import Literal
from dataclasses import dataclass
import numpy as np
from numpy import cosh, sinh, cos, pi
from numpy.typing import NDArray

from bipedal_floating_description.utils.config import Config, Stance, FtScalar, Ft, End

m = Config.MASS
H = Config.IDEAL_Z_COM_IN_WF
Hpel = Config.IDEAL_Z_PEL_IN_WF
h = Config.STEP_HEIGHT
w = Config.OMEGA
W = Config.IDEAL_Y_STEPLENGTH

Ts = Config.Ts
NL = Config.NL_MARCHINPLACE
T = NL * Ts

AlipScalarData = dict[Literal['lf', 'rf', 'com', 'cfTOcom', 'L'], float]

class AlipSfHeightFitting:...
    
@dataclass
class AbstractAlipTrajOneDirection:
    stance: Stance
    Tn: int
    ft0: FtScalar
    com0: float
    var0: NDArray
    
    def __post_init__(self):
        self.AT = self._A(T)
    
    @staticmethod
    def _A(t: float) -> NDArray:...
    def _Ldes_2T(self) -> float:...
    
    def plan(self) -> AlipScalarData:
        cf, sf = self.stance
        t = self.Tn * Ts
        
        # 下一點質心軌跡(相對支撐腳)
        cfTOcom1, L1 = self._A(t) @ self.var0
        
        # 下一步估測的角動量
        Le_1T = self._A(T)[1,:] @ self.var0
        
        # 下下步所欲的角動量
        Ldes_2T = self._Ldes_2T()
        
        # 下一步的擺動腳落地點
        sf2com_T = (Ldes_2T - self.AT[1,1]*Le_1T) / self.AT[1,0]
        
        # 擬合出下一點擺動腳軌跡
        ratio = Tn / NL
        sfTOcom1 = 0.5 * ( (1+cos(pi*ratio))*(self.com0 - self.ft0[sf]) + (1-cos(pi*ratio))*sf2com_T )
        
        return {
            cf: self.ft0[cf],
            'com': self.ft0[cf] + cfTOcom1,
            sf: self.ft0[cf] + cfTOcom1 - sfTOcom1,
            'cfTOcom': cfTOcom1,
            'L': L1
        }

class AlipTrajY(AbstractAlipTrajOneDirection):
    @staticmethod
    def _A(t: float) -> NDArray:
        return np.array([
            [          cosh(w*t), -sinh(w*t)/(m*H*w) ],
            [ -m*H*w * sinh(w*t),  cosh(w*t)         ]
        ])
    def _Ldes_2T(self) -> float:
        cf, sf = self.stance
        sign = {'lf': -1, 'rf': 1}[cf]
        return sign * 0.5 *m*H*W * w*sinh(w*T) / (1+cosh(w*T))

class Plan:
    pel0: NDArray
    lf0: NDArray
    rf0: NDArray
    com0: NDArray

    def __init__(self, stance: Stance, Tn: int, vary0: NDArray):
        self.stance = stance
        self.Tn = Tn
        self.vary0 = vary0
        
    @classmethod
    def initilize(cls, end_in_wf: End, com0: NDArray):
        cls.pel0 = end_in_wf['pel']
        cls.lf0 = end_in_wf['lf']
        cls.rf0 = end_in_wf['rf']
        cls.com0 = com0
        
    def plan(self):
        y0_ft: FtScalar = {
            'rf': self.lf0[1],
            'lf': self.rf0[1]
        }
        datay = AlipTrajY(self.stance, self.Tn, y0_ft, self.com0[1], self.vary0).plan()
        
        ref_p_end: End = {
            'lf': np.array([0, datay['lf']])
        }