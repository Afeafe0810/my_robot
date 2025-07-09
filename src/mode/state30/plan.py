from typing import Literal
from dataclasses import dataclass

import numpy as np
from numpy import cosh, sinh, cos, pi
from numpy.typing import NDArray

from src.utils.config import Config, Stance, FtScalar, Ft, End

storage: list = []

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

def alipSfHeight_fitting(Tn: int):
    if Tn < NL/2:
        return h*1.8
    else:
        return h - 4*h*(Tn/NL - 0.5)**2
        
    
@dataclass
class AbstractAlipTrajOneDirection:
    stance: Stance
    Tn: int
    ft0: FtScalar
    com0: float
    L0: float
    
    def __post_init__(self):
        self.AT = self._A(T)
        self.cfTOcom0 = self.com0 - self.ft0[self.stance.cf]
        self.sfTOcom0 = self.com0 - self.ft0[self.stance.sf]
        self.var0 = np.array([self.cfTOcom0, self.L0])
    
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
        ratio = self.Tn / NL
        sfTOcom1 = 0.5 * ( (1+cos(pi*ratio))*(self.sfTOcom0) + (1-cos(pi*ratio))*sf2com_T )
        
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
        sign = {'lf': -1, 'rf': 1}[self.stance.cf]
        return sign * 0.5 *m*H*W * w*sinh(w*T) / (1+cosh(w*T))

class Plan:
    pel0: NDArray
    lf0: NDArray
    rf0: NDArray
    com0: NDArray
    Ly0: float
    Lx0: float

    def __init__(self, stance: Stance, Tn: int, com: NDArray, pel: NDArray):
        self.stance = stance
        self.Tn = Tn
        self.com = com
        self.pel = pel
        
    @classmethod
    def initilize(cls, end0_in_wf: End, com0: NDArray, Ly0: float, Lx0: float):
        cls.pel0 = end0_in_wf['pel']
        cls.lf0 = end0_in_wf['lf']
        cls.rf0 = end0_in_wf['rf']
        cls.com0 = com0
        cls.Ly0 = Ly0
        cls.Lx0 = Lx0
        
        
    def plan(self) -> tuple[End, End, NDArray, NDArray]:
        cf, sf = self.stance
        
        y0_ft: FtScalar = {'lf': self.lf0[1], 'rf': self.rf0[1]}
        datay = AlipTrajY(self.stance, self.Tn, y0_ft, self.com0[1], self.Lx0).plan()
        
        x0_cf = {'lf': self.lf0, 'rf': self.rf0}[cf][0]
        y_pel = datay['com'] - self.com[1] + self.pel[1]
        z_sf = alipSfHeight_fitting(self.Tn)
        
        ref_p_end: End = {
            cf: np.array([x0_cf, datay[cf], 0]),
            sf: np.array([x0_cf, datay[sf], z_sf]),
            'pel': np.array([x0_cf, y_pel, Hpel])
        }
        
        ref_a_end: End = {
            'lf': np.zeros(3),
            'rf': np.zeros(3),
            'pel': np.zeros(3)
        }
        ref_varx = np.zeros(2)
        ref_vary = np.array([datay['cfTOcom'], datay['L']])
        
        storage.append({**datay, 'z_sf': z_sf})
        return ref_p_end, ref_a_end, ref_varx, ref_vary
    
if __name__ == '__main__':
    if NL == 50:
        com0 = 0.08
    elif NL == 25:
        com0 = 0.05
        
    ft0: FtScalar = {'lf': 0.1, 'rf': -0.1}
    stance = Stance('lf', 'rf')
    L0 = 0
    for Tn in range(0, NL+1):
        alip = AlipTrajY(
            stance,
            Tn,
            ft0,
            com0,
            L0
        )
        data = alip.plan()
        storage.append(data)
        
        

    stance = Stance(stance.sf, stance.cf)
    ft0 = {'lf': data['lf'], 'rf': data['rf']}
    com0 = data['com']
    L0 = data['L']
    for Tn in range(1, NL+1):
        alip = AlipTrajY(
            stance,
            Tn,
            ft0,
            com0,
            L0
        )
        data = alip.plan()
        storage.append(data)
    
    stance = Stance(stance.sf, stance.cf)
    ft0 = {'lf': data['lf'], 'rf': data['rf']}
    com0 = data['com']
    L0 = data['L']
    for Tn in range(1, NL+1):
        alip = AlipTrajY(
            stance,
            Tn,
            ft0,
            com0,
            L0
        )
        data = alip.plan()
        storage.append(data)

    # stance = Stance(stance.sf, stance.cf)
    # ft0 = {'lf': data['lf'], 'rf': data['rf']}
    # com0 = data['com']
    # L0 = data['L']
    # for Tn in range(1, NL+1):
    #     alip = AlipTrajY(
    #         stance,
    #         Tn,
    #         ft0,
    #         com0,
    #         L0
    #     )
    #     data = alip.plan()
    #     storage.append(data)
    
    # stance = Stance(stance.sf, stance.cf)
    # ft0 = {'lf': data['lf'], 'rf': data['rf']}
    # com0 = data['com']
    # L0 = data['L']
    # for Tn in range(1, NL+1):
    #     alip = AlipTrajY(
    #         stance,
    #         Tn,
    #         ft0,
    #         com0,
    #         L0
    #     )
    #     data = alip.plan()
    #     storage.append(data)
    
    # stance = Stance(stance.sf, stance.cf)
    # ft0 = {'lf': data['lf'], 'rf': data['rf']}
    # com0 = data['com']
    # L0 = data['L']
    # for Tn in range(1, NL+1):
    #     alip = AlipTrajY(
    #         stance,
    #         Tn,
    #         ft0,
    #         com0,
    #         L0
    #     )
    #     data = alip.plan()
    #     storage.append(data)
    
    import pandas as pd
    import os
    pd.DataFrame(storage).to_csv(os.path.join(Config.DIR_OUTPUT, 'Test.csv'))
