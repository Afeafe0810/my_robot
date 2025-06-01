import rclpy; from rclpy.node import Node
from std_msgs.msg import Float64MultiArray 

import numpy as np; np.set_printoptions(precision=2)
import pandas as pd
from copy import deepcopy
#================ import other code =====================#
from utils.config import Config
from utils.ros_interfaces import ROSInterfaces as ROS
from utils.robot_model import RobotModel
from utils.frame_kinermatic import RobotFrame
from motion_planning import Trajatory
from torque_control import TorqueControl
#========================================================#
"""
#TODO 
    1. 根據state設計物件導向：因為這樣每次擴充功能都要找if子句，太容易錯了
        main: set_stance
        robot model gravity
        motion planning
        knee control
    2. urdf檔的initial angle
"""
class UpperLevelController(Node):

    def __init__(self):
        #建立ROS的node
        super().__init__('upper_level_controllers')
        
        #負責ROS的功能
        ROS.init(self, self.main_controller_callback)
        
        #機器人的模型
        self.robot = RobotModel()
                
        #負責量測各部位的位置與姿態
        self.frame = RobotFrame()
        
        #負責處理軌跡
        self.traj = Trajatory()
        
        #負責處理扭矩
        self.ctrl = TorqueControl()
        
        #============機器人的重要參數=====================#     
        
        #主被動腳
        self.stance : list[str] = ['lf', 'rf'] #第一個是支撐腳cf, 第二個是擺動腳sf
        self.stance_past : list[str] = ['lf', 'rf'] #上個取樣時間的支撐腳與擺動腳
        
        self.ref_record     = pd.DataFrame(columns = Config.ALIP_COLUMN_TITLE)
        self.measure_record = pd.DataFrame(columns = Config.ALIP_COLUMN_TITLE)
 
    def main_controller_callback(self):
        #==========拿取訂閱值==========#
        p_base_in_wf, r_base_to_wf, state, is_contact, jp, jv, force_ft, tau_ft = ROS.subscribers.return_data()
        
        #==========更新可視化的機器人==========#
        config = self.robot.update_VizAndMesh(jp)
        
        #==========更新frame==========#
        self.frame.updateFrame(self.robot, config, p_base_in_wf, r_base_to_wf, jp)        
        
        #========接觸判斷========#
        # is_firmly = self._is_step_firmly(force_ft)
        is_firmly = self._is_no_hight(self.frame)
        
        #========支撐狀態切換=====#
        self._set_stance(state)
        
        #========軌跡規劃========#
        ref = self.traj.plan(state, self.frame, self.stance, is_firmly)
        if state == 30:
            self.ref_record = ref.to_csv(self.ref_record)
            self.measure_record = self.frame.to_csv(self.measure_record, self.stance)
            
        #========扭矩控制========#
        torque = self.ctrl.update_torque(self.frame, self.robot, ref, state, self.stance, self.stance_past, is_firmly, jp, jv)
        ROS.publishers.effort.publish(torque)
        self.stance_past = self.stance

    def _set_stance(self, state):
        """掌管state 0, 1, 2的支撐腳邏輯"""
        if state in [0, 1, 2, 3]:
            #(這邊不能用return寫，否則state30會是None)
            self.stance = ['lf', 'rf']
        if state == 30:
            self.stance = self.traj.aliptraj.stance
            
    @staticmethod
    def _is_step_firmly(force_ft: dict[str, float]) -> dict[str, bool]:
        """
        用力規讀取z方向的分力, 來判斷擺動腳是否踩穩, 若受力 > 10N則視為踩穩
        (這對擺動腳比較重要)
        """
        threshold = 10
        return { key: val> threshold for key, val in force_ft.items()}
    
    @staticmethod
    def _is_no_hight(frame: RobotFrame) -> dict[str, bool]:
        return {
            'lf': frame.p_lf_in_wf[2, 0] <= 0.01,
            'rf': frame.p_rf_in_wf[2, 0] <= 0.01,
        }

def main(args=None):
    rclpy.init(args=args)

    upper_level_controllers = UpperLevelController()
    rclpy.spin(upper_level_controllers)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    upper_level_controllers.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
