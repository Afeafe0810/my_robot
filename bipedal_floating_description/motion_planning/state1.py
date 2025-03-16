import numpy as np; np.set_printoptions(precision=2)
#================ import library ========================#
from motion_planning.utils import Ref
#========================================================#

# state 0, 1
def bipedalBalance_plan():
    """回傳雙腳支撐時的參考值"""
    return Ref(
        pel := np.vstack(( 0,    0, 0.55, 0, 0, 0 )),
        lf  := np.vstack(( 0,  0.1,    0, 0, 0, 0 )),
        rf   = np.vstack(( 0, -0.1,    0, 0, 0, 0 )),
        var  = Ref._generate_var_insteadOf_com(pel, lf)
    )
