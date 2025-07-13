import os
from copy import deepcopy

import numpy as np; np.set_printoptions(precision=2)
import pandas as pd

import rclpy; from rclpy.node import Node
from std_msgs.msg import Float64MultiArray 

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
from src.mode.state5 import State5
from src.mode.state30.state30 import State30
import src.mode.state30.plan as plan

#========================================================#

measure: list = []

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
        
        self.state0 = State0()
        self.state1 = State1()

 
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
        
        
        model_gravity = self.robot.gravity(jp)
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
        jv_ft : Ft = {'lf': jv[:6, 0], 'rf': jv[6:, 0]}
        jp_ft : Ft = {'lf': jp[:6, 0], 'rf': jp[6:, 0]}
        
        print(f"===== state{state} =====")
        match state:
            case 0:
                torque = self.state0.ctrl(
                    jp.flatten(),
                    model_gravity,
                    p_end_in_pf
                )
            case 1:
                torque = self.state1.ctrl(
                    p_end_in_wf,
                    p_end_in_pf,
                    a_end_in_pf,
                    model_gravity,
                    jp_ft,
                    jv_ft,
                    eularToGeo,
                    J
                )
        
                print("Zpel: ", self.frame.p_pel_in_wf[2,0])
                
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
                print("Zpel: ", self.frame.p_pel_in_wf[2,0])
                
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
                print("Zpel: ", self.frame.p_pel_in_wf[2,0])
                
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
                print("Zpel: ", self.frame.p_pel_in_wf[2,0])

            case 5:
                torque = State5(
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
                print("Zpel: ", self.frame.p_pel_in_wf[2,0])
                print("Y_cfTOcom: ", self.frame.p_com_in_wf[1,0] - self.frame.p_lf_in_wf[1,0])
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
                self.save(measure)
                
                
                
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

    def save(self, storage: list):
        key = ('z_pel', 'x_lf', 'y_lf', 'z_lf', 'x_rf', 'y_rf', 'z_rf', 'x_com', 'y_com', 'z_com', 'Ly', 'Lx')
        value = (
            self.frame.p_pel_in_wf[2, 0],
            *self.frame.p_lf_in_wf[:,0],
            *self.frame.p_rf_in_wf[:,0],
            *self.frame.p_com_in_wf[:,0],
            self.frame.L_com_in_lf['y'], self.frame.L_com_in_lf['y']
        )
        data = dict(zip(key, value))
        storage.append(data)
        
        
def main(args=None):
    rclpy.init(args=args)
    upper_level_controllers = UpperLevelController()
    
    try:
        rclpy.spin(upper_level_controllers)
        
    except KeyboardInterrupt:
        print('\nKeyboard Interrupt')
        
        pd.DataFrame(plan.storage).to_csv(os.path.join(Config.DIR_OUTPUT, 'storage30.csv'))
        pd.DataFrame(measure).to_csv(os.path.join(Config.DIR_OUTPUT, 'measure30.csv'))
        print('save success')
    
    finally:
        upper_level_controllers.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
