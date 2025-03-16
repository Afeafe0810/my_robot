#================ import library ========================#
from utils.frame_kinermatic import RobotFrame
from motion_planning.utils import Ref
from motion_planning.state1 import bipedalBalance_plan
from motion_planning.state2 import LeftLegBalance
from motion_planning.state30 import AlipTraj
#========================================================#
class Trajatory:
    def __init__(self):
        #state 2
        self.lf_stand = LeftLegBalance()
        #state 30
        self.aliptraj = AlipTraj()
    
    def plan(self, state: float, frame: RobotFrame, stance: list[str], is_firmly: dict[str, bool]) -> Ref:
        match state:          
            case 0 | 1 : #雙支撐
                return bipedalBalance_plan()

            case 2: #左腳站立
                return self.lf_stand.plan()
                     
            case 30:#步行
                return self.aliptraj.plan(frame, 0, is_firmly, stance)