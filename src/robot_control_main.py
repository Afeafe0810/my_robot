import rclpy; from rclpy.node import Node
from std_msgs.msg import Float64MultiArray 

import numpy as np; np.set_printoptions(precision=2)
import pandas as pd
from copy import deepcopy
#================ import other code =====================#
from src.utils.config import Config, End, Ft
from src.utils.ros_interfaces import ROSInterfaces as ROS
from src.utils.robot_model import RobotModel
from src.utils.frame_kinermatic import RobotFrame
from src.motion_planning import Trajatory
from src.torque_control import TorqueControl

from src.mode.state0 import State0
from src.mode.state1 import State1
from src.mode.state2 import State2
from src.mode.state3 import State3
from src.mode.state4 import State4
from src.mode.state30.state30 import State30

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
        
        
        model_gravity = self.robot.new_gravity(jp)
        p_end_in_pf: End = {
            'lf': self.frame.p_lf_in_pf.flatten(),
            'rf': self.frame.p_rf_in_pf.flatten(),
            'pel': self.frame.p_pel_in_pf.flatten()
        }
        a_end_in_pf: End = {
            'lf': self.frame.pa_lf_in_pf[3:, 0],
            'rf': self.frame.pa_rf_in_pf[3:, 0],
            'pel': self.frame.pa_pel_in_pf[3:, 0]
        }
        p_end_in_wf: End = {
            'lf': self.frame.p_lf_in_wf.flatten(),
            'rf': self.frame.p_rf_in_wf.flatten(),
            'pel': self.frame.p_pel_in_wf.flatten()
        }
        J: Ft = {ft: compnt for ft, compnt in zip( ('lf','rf'), self.frame.get_jacobian() )}
        eularToGeo: Ft = self.frame.eularToGeo
        
        match state:
            case 0:
                torque = State0(jp.flatten(), model_gravity, p_end_in_pf).ctrl()
            case 1:
                torque = State1(
                    p_end_in_wf,
                    p_end_in_pf,
                    a_end_in_pf,
                    model_gravity,
                    jp.flatten(),
                    jv.flatten(),
                    eularToGeo,
                    J
                ).ctrl()
            case 2:
                torque = State2(
                    p_end_in_wf,
                    p_end_in_pf,
                    a_end_in_pf,
                    model_gravity,
                    jp.flatten(),
                    jv.flatten(),
                    eularToGeo,
                    J
                ).ctrl()
            case 3:
                torque = State3(
                    p_end_in_wf,
                    p_end_in_pf,
                    a_end_in_pf,
                    model_gravity,
                    jp.flatten(),
                    jv.flatten(),
                    eularToGeo,
                    J
                ).ctrl()
            case 4:
                torque = State4(
                    p_end_in_wf,
                    p_end_in_pf,
                    a_end_in_pf,
                    self.frame.p_com_in_wf.flatten(),
                    self.frame.L_com_in_lf['y'],
                    self.frame.L_com_in_lf['x'],
                    model_gravity,
                    jp.flatten(),
                    jv.flatten(),
                    eularToGeo,
                    J
                ).ctrl()
            case 30:
                torque = State30(
                    p_end_in_wf,
                    p_end_in_pf,
                    a_end_in_pf,
                    self.frame.p_com_in_wf.flatten(),
                    self.frame.L_com_in_lf['y'],
                    self.frame.L_com_in_lf['x'],
                    model_gravity,
                    jp.flatten(),
                    jv.flatten(),
                    eularToGeo,
                    J
                ).ctrl()
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
    
    try:
        rclpy.spin(upper_level_controllers)
        
    except KeyboardInterrupt:
        print('\nKeyboard Interrupt')
    
    finally:
        upper_level_controllers.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
