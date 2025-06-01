#================ import library ========================#
from utils.frame_kinermatic import RobotFrame
from motion_planning.utils import Ref
from motion_planning.state1 import BipedalBalance
from motion_planning.state2 import LeftLegBalance
from motion_planning.state3 import FancyStand
from motion_planning.state30 import AlipTraj
#========================================================#
class Trajatory:
    def __init__(self):
        #state 1
        self.bipedalBalance = BipedalBalance()
        #state 2
        self.lf_stand = LeftLegBalance()
        # state 3
        self.fancy_stand = FancyStand()
        #state 30
        self.aliptraj = AlipTraj()
    
    def plan(self, state: float, frame: RobotFrame, stance: list[str], is_firmly: dict[str, bool]) -> Ref:
        match state:          
            case 0 : #雙支撐
                return None
            
            case 1 : #雙支撐
                return self.bipedalBalance.plan(frame)

            case 2: #左腳站立
                return self.lf_stand.plan()
            
            case 3: #花式擺腿
                return self.fancy_stand.plan()
                     
            case 30:#步行
                return self.aliptraj.plan(frame, 0, is_firmly, stance)