import numpy as np; np.set_printoptions(precision=2)
from numpy.typing import NDArray
from typing import Literal
#================ import library ========================#
from bipedal_floating_description.utils.robot_model import RobotModel
#========================================================#

class State0:
    def __init__(self, jp: NDArray, model_gravity: dict[str, NDArray], end_in_pf: dict[Literal['lf', 'rf', 'pel'], NDArray]):
        self.jp = jp.flatten()
        self.tauG = self.gravity(model_gravity, end_in_pf)
        
        self.ref_jp = np.zeros(12)
        self.kp = np.array([ 2, 2, 4, 6, 6, 4 ]*2)

    def gravity(self, model_gravity: dict[str, NDArray], end_in_pf: dict[Literal['lf', 'rf', 'pel'], NDArray]) -> NDArray:
        # 根據骨盆位置來判斷重心腳
        end = end_in_pf
        y_ftTOpel = {
            'lf': abs(end['lf'][1] - end['pel'][1]),
            'rf': abs(end['rf'][1] - end['pel'][1])
        }
        
        gf = 'lf' if y_ftTOpel['lf'] <= y_ftTOpel['rf'] else 'rf'
        
        # 再根據重心腳距離權重
        return model_gravity[gf] * (1 - y_ftTOpel[gf]/0.1) + model_gravity['from_both_single_ft'] * (y_ftTOpel[gf]/0.1)
            
    def ctrl(self):
        return self.kp * (self.ref_jp - self.jp) + self.tauG