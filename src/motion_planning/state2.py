import numpy as np; np.set_printoptions(precision=2)
#================ import library ========================#
from src.motion_planning.utils import Ref
from src.utils.config import Config
#========================================================#

# state 2
class LeftLegBalance:
    """回傳左腳支撐時的參考值"""
    def __init__(self):
        self.t : float = 0.0
        
    def plan(self):
        T = Config.DDT
        Ts = Config.Ts
                        
        #==========線性移動==========#
        def linearMove(x0, x1, t0, t1):
            return np.clip(x0 + (x1-x0) * (self.t-t0)/(t1-t0), x0, x1 )
            
        y_pel = linearMove(*[0, 0.09], *[0*T, 0.5*T])
        # z_sf  = linearMove(*[0, Config.STEP_HEIGHT], *[1*T, 1.1*T])
        z_sf  = 0.0
        angle = linearMove(0, 0.14973, 0, 0.5*T)
        
        if self.t < 2 * T:
            print(f"LeftLegBalance.t = {self.t:.2f}")
            self.t += Ts            

        return Ref(
            pel := np.vstack(( 0, y_pel, 0.55, 0, 0, 0 )),
            lf  := np.vstack(( 0,   0.1,    0, 0, 0, 0 )),
            rf   = np.vstack(( 0,  -0.1, z_sf, 0, 0, 0 )),
            var  = Ref._generate_var_insteadOf_com(pel, lf),
            ax = angle
        )
