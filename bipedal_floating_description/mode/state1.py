import numpy as np; np.set_printoptions(precision=2)
from numpy.typing import NDArray
from typing import Literal
#================ import library ========================#
from bipedal_floating_description.utils.robot_model import RobotModel
from bipedal_floating_description.mode.utils import linear_move
from bipedal_floating_description.utils.config import Config
#========================================================#

Hpel = Config.IDEAL_Z_PEL_IN_WF
NL = Config.TL_BALANCE

class State1:
    stance: list[str] = ['lf', 'rf']
    Tn: int = 0 ~~要更新時間 # TODO
    is_just_started: bool = True ~~要記得關閉 #TODO
    pel0: NDArray
    lf0: NDArray
    rf0: NDArray
    
    def __init__(self, end_in_wf: dict[Literal['lf', 'rf', 'pel'], NDArray]):
        cls = self.__class__
        if cls.is_just_started:
            cls.pel0 = end_in_wf['pel']
            cls.lf0 = end_in_wf['lf']
            cls.rf0 = end_in_wf['rf']

    @staticmethod
    def gravity(model_gravity: dict[str, NDArray], end_in_pf: dict[Literal['lf', 'rf', 'pel'], NDArray]) -> NDArray:
        # 根據骨盆位置來判斷重心腳
        end = end_in_pf
        y_ftTOpel = {
            'lf': abs(end['lf'][1] - end['pel'][1]),
            'rf': abs(end['rf'][1] - end['pel'][1])
        }
        
        gf = 'lf' if y_ftTOpel['lf'] <= y_ftTOpel['rf'] else 'rf'
        
        # 再根據重心腳距離權重
        return model_gravity[gf] * (1 - y_ftTOpel[gf]/0.1) + model_gravity['from_both_single_ft'] * (y_ftTOpel[gf]/0.1)
    
    def plan(self):
        z_pel = linear_move(self.Tn, 0, NL, self.pel0[2], Hpel)
        return {
            'p_pel': np.hstack((self.pel0[:2], z_pel)),
            'p_lf': self.lf0,
            'p_rf': self.rf0
        }
    
    def ctrl(self) -> NDArray:...
        