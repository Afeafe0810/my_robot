from std_msgs.msg import Float64MultiArray 
import numpy as np; np.set_printoptions(precision=2)
from copy import deepcopy
#================ import other code =====================#
from utils.config import Config
#========================================================#

class RobotFrame:
    def __init__(self):
        self.p_base_in_wf: np.ndarray = None
        self.r_base_to_wf: np.ndarray = None
        
        pass

