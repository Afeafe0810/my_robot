import numpy as np; np.set_printoptions(precision=2)
from numpy.typing import NDArray

#================ import library ========================#
from src.utils.robot_model import RobotModel
from src.utils.config import GravityDict, End
#========================================================#


def gravity(model_gravity: GravityDict, end_in_pf: End) -> NDArray:
    # 根據骨盆位置來判斷重心腳
    end = end_in_pf
    y_ftTOpel = {
        'lf': abs(end['lf'][1] - end['pel'][1]),
        'rf': abs(end['rf'][1] - end['pel'][1])
    }
    
    gf = 'lf' if y_ftTOpel['lf'] <= y_ftTOpel['rf'] else 'rf'
    
    # 再根據重心腳距離權重
    return model_gravity[gf] * (1 - y_ftTOpel[gf]/0.1) + model_gravity['from_both_single_ft'] * (y_ftTOpel[gf]/0.1)

class State0:
    """關節角度的單環回授, 用於剛啟動時維持平衡"""
    
    ref_jp = np.zeros(12)
    kp = np.array([ 2, 2, 4, 6, 6, 4 ]*2)
    
    def ctrl(
        self,
        jp: NDArray,
        model_gravity: GravityDict,
        end_in_pf: End
        
    ) -> NDArray:

        tauG = gravity(model_gravity, end_in_pf)
        
        return self.kp * (self.ref_jp - jp) + tauG
        