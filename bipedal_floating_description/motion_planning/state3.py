import numpy as np; np.set_printoptions(precision=2)
#================ import library ========================#
from motion_planning.utils import Ref
from utils.config import Config
#========================================================#

# state 2
class FancyStand:
    """回傳左腳支撐時的參考值"""
    def __init__(self):
        self.t : float = 0.0
        
    def plan(self):
        T = Config.DDT
        Ts = Config.Ts
                        
        #==========線性移動==========#
        def linearMove(x0, x1, t0, t1):
            return np.clip(x0 + (x1-x0) * (self.t-t0)/(t1-t0), x0, x1 )
            
        z_sf  = linearMove(*[0, 0.05], *[1*T, 1.5*T])
        # z_sf  = 0.0
        
        if self.t < 2 * T:
            print(f"LeftLegBalance.t = {self.t:.2f}")
            self.t += Ts

        return Ref(
            pel := np.vstack(( 0, 0.09, 0.55, 0, 0, 0 )),
            lf  := np.vstack(( 0,   0.1,    0, 0, 0, 0 )),
            rf   = np.vstack(( 0,  -0.1, z_sf, 0, 0, 0 )),
            var  = Ref._generate_var_insteadOf_com(pel, lf),
        )
    # def plan(self):

    #     return Ref(
    #         pel := np.vstack(( 0, 0.08, 0.55, 0, 0, 0 )),
    #         lf  := np.vstack(( 0,   0.1,    0, 0, 0, 0 )),
    #         rf   = np.vstack(( 0,  -0.1,    0, 0, 0, 0 )),
    #         var  = Ref._generate_var_insteadOf_com(pel, lf),
    #     )