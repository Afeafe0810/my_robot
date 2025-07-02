import rclpy; from rclpy.node import Node
from std_msgs.msg import Float64MultiArray 

import numpy as np; np.set_printoptions(precision=2)
import pandas as pd
from copy import deepcopy
#================ import other code =====================#
from bipedal_floating_description.utils.config import Config, End
from bipedal_floating_description.utils.ros_interfaces import ROSInterfaces as ROS
from bipedal_floating_description.utils.robot_model import RobotModel
from bipedal_floating_description.utils.frame_kinermatic import RobotFrame
from bipedal_floating_description.motion_planning import Trajatory
from bipedal_floating_description.torque_control import TorqueControl

from bipedal_floating_description.mode.state0 import State0
from bipedal_floating_description.mode.state1 import State1
from bipedal_floating_description.mode.state2 import State2
from bipedal_floating_description.mode.state3 import State3

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
        self.export_data_for_senior = []
 
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
        
        match state:
            case 0:
                model_gravity = self.robot.new_gravity(jp)
                end_in_pf: End = {
                    'lf': self.frame.p_lf_in_pf.flatten(),
                    'rf': self.frame.p_rf_in_pf.flatten(),
                    'pel': self.frame.p_pel_in_pf.flatten()
                }
                torque = State0(jp.flatten(), model_gravity, end_in_pf).ctrl()
            case 1:
                Jlf, Jrf = self.frame.get_jacobian()
                torque = State1(
                    {'lf': self.frame.p_lf_in_wf.flatten(), 'rf': self.frame.p_rf_in_wf.flatten(), 'pel': self.frame.p_pel_in_wf.flatten()},
                    {'lf': self.frame.p_lf_in_pf.flatten(), 'rf': self.frame.p_rf_in_pf.flatten(), 'pel': self.frame.p_pel_in_pf.flatten()},
                    {'lf': self.frame.pa_lf_in_pf[3:,0], 'rf': self.frame.pa_rf_in_pf[0], 'pel': self.frame.pa_pel_in_pf[0]},
                    self.robot.new_gravity(jp),
                    jp.flatten(),
                    jv.flatten(),
                    self.frame.eularToGeo,
                    {'lf': Jlf, 'rf': Jrf}
                ).ctrl()
            case 2:
                Jlf, Jrf = self.frame.get_jacobian()
                torque = State2(
                    {'lf': self.frame.p_lf_in_wf.flatten(), 'rf': self.frame.p_rf_in_wf.flatten(), 'pel': self.frame.p_pel_in_wf.flatten()},
                    {'lf': self.frame.p_lf_in_pf.flatten(), 'rf': self.frame.p_rf_in_pf.flatten(), 'pel': self.frame.p_pel_in_pf.flatten()},
                    {'lf': self.frame.pa_lf_in_pf[3:,0], 'rf': self.frame.pa_rf_in_pf[0], 'pel': self.frame.pa_pel_in_pf[0]},
                    self.robot.new_gravity(jp),
                    jp.flatten(),
                    jv.flatten(),
                    self.frame.eularToGeo,
                    {'lf': Jlf, 'rf': Jrf}
                ).ctrl()
            case 3:
                Jlf, Jrf = self.frame.get_jacobian()
                torque = State3(
                    {'lf': self.frame.p_lf_in_wf.flatten(), 'rf': self.frame.p_rf_in_wf.flatten(), 'pel': self.frame.p_pel_in_wf.flatten()},
                    {'lf': self.frame.p_lf_in_pf.flatten(), 'rf': self.frame.p_rf_in_pf.flatten(), 'pel': self.frame.p_pel_in_pf.flatten()},
                    {'lf': self.frame.pa_lf_in_pf[3:,0], 'rf': self.frame.pa_rf_in_pf[0], 'pel': self.frame.pa_pel_in_pf[0]},
                    self.robot.new_gravity(jp),
                    jp.flatten(),
                    jv.flatten(),
                    self.frame.eularToGeo,
                    {'lf': Jlf, 'rf': Jrf}
                ).ctrl()
                
            case _:        
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
    def get_data(self, storage: list, torque: np.ndarray, jp: np.ndarray, jv: np.ndarray, tauG: np.ndarray):
        data = {
            'x_pel': self.frame.p_pel_in_wf[0, 0],
            'y_pel': self.frame.p_pel_in_wf[1, 0],
            'z_pel': self.frame.p_pel_in_wf[2, 0],
            
            'tau0': torque[0, 0],
            'tau1': torque[1, 0],
            'tau2': torque[2, 0],
            'tau3': torque[3, 0],
            'tau4': torque[4, 0],
            'tau5': torque[5, 0],
            'tau6': torque[6, 0],
            'tau7': torque[7, 0],
            'tau8': torque[8, 0],
            'tau9': torque[9, 0],
            'tau10': torque[10, 0],
            'tau11': torque[11, 0],
            
            'jp0': jp[0, 0],
            'jp1': jp[1, 0],
            'jp2': jp[2, 0],
            'jp3': jp[3, 0],
            'jp4': jp[4, 0],
            'jp5': jp[5, 0],
            'jp6': jp[6, 0],
            'jp7': jp[7, 0],
            'jp8': jp[8, 0],
            'jp9': jp[9, 0],
            'jp10': jp[10, 0],
            'jp11': jp[11, 0],
            
            'jv0': jv[0, 0],
            'jv1': jv[1, 0],
            'jv2': jv[2, 0],
            'jv3': jv[3, 0],
            'jv4': jv[4, 0],
            'jv5': jv[5, 0],
            'jv6': jv[6, 0],
            'jv7': jv[7, 0],
            'jv8': jv[8, 0],
            'jv9': jv[9, 0],
            'jv10': jv[10, 0],
            'jv11': jv[11, 0],
            
            
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
