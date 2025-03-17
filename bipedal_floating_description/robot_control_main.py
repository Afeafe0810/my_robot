import rclpy; from rclpy.node import Node
from std_msgs.msg import Float64MultiArray 

import numpy as np; np.set_printoptions(precision=2)
import pandas as pd

#================ import other code =====================#
from utils.config import Config
from utils.ros_interfaces import ROSInterfaces, RobotModel
from utils.frame_kinermatic import RobotFrame
from utils.motion_planning import *
from utils.torque_control import *
#========================================================#

class UpperLevelController(Node):

    def __init__(self):
        #建立ROS的node
        super().__init__('upper_level_controllers')
        
        #負責ROS的功能
        self.ros = ROSInterfaces(self, self.main_controller_callback)
        
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
        
        self.ref_store = pd.DataFrame(columns=[
            'pel_x', 'pel_y', 'pel_z',
            'lf_x', 'lf_y', 'lf_z',
            'rf_x', 'rf_y', 'rf_z',
            'x', 'y',
            'Ly', 'Lx'
        ])
        self.mea_store = deepcopy(self.ref_store)
 
    def main_controller_callback(self):
        #==========拿取訂閱值==========#
        p_base_in_wf, r_base_to_wf, state, is_contact, jp, jv, force_ft, tau_ft = self.ros.returnSubData()
        
        #==========更新可視化的機器人==========#
        config = self.robot.update_VizAndMesh(jp)
        
        #==========更新frame==========#
        self.frame.updateFrame(self.robot, config, p_base_in_wf, r_base_to_wf, jp)        
        
        #========接觸判斷========#
        #[ ] 用高度來進行接觸判斷
        # is_firmly = self._is_step_firmly(force_ft)
        is_firmly = self._is_no_hight(self.frame)
        
        #========支撐狀態切換=====#
        self._set_stance(state)
        
        #========軌跡規劃========#
        ref = self.traj.plan(state, self.frame, self.stance, is_firmly)
        if state == 30 and not ref.need_push:
            self.ref_store = Test.store_ref(self.ref_store, ref)
            self.mea_store = Test.store_mea(self.mea_store, self.frame, self.stance)
            self.ref_store.to_csv("real_planning.csv")
            self.mea_store.to_csv("real_measure.csv")
            
            
        #========扭矩控制========#
        torque = self.ctrl.update_torque(self.frame, self.robot, ref, state, self.stance, self.stance_past, is_firmly, jp, jv)
        self.ros.publisher['effort'].publish( Float64MultiArray(data = torque) )
        #TODO ALIP計數要加
        self.stance_past = self.stance

    def _set_stance(self, state):
        """掌管state 0, 1, 2的支撐腳邏輯"""
        if state in [0, 1, 2]:
            #(這邊不能用return寫，否則state30會是None)
            self.stance = ['lf', 'rf']
        #TODO 用ALIP計數
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

class Test:
    @staticmethod
    def store_ref(ref_store: pd.DataFrame, ref : Ref):
        new_data = {
            'pel_x': ref.pel[0, 0],
            'pel_y': ref.pel[1, 0],
            'pel_z': ref.pel[2, 0],

            'lf_x': ref.lf[0, 0],
            'lf_y': ref.lf[1, 0],
            'lf_z': ref.lf[2, 0],

            'rf_x': ref.rf[0, 0],
            'rf_y': ref.rf[1, 0],
            'rf_z': ref.rf[2, 0],

            'x': ref.var['x'][0, 0],
            'y': ref.var['y'][0, 0],
            
            'Ly': ref.var['x'][1, 0],
            'Lx': ref.var['y'][1, 0],
        }
        # 建立一筆資料的 DataFrame
        new_df = pd.DataFrame([new_data])
        # 使用 pd.concat 進行疊加
        ref_store = pd.concat([ref_store, new_df], ignore_index=True)
        return ref_store

    @staticmethod
    def store_mea(mea_store: pd.DataFrame, frame: RobotFrame, stance: list[str]):
        var = frame.get_alipdata(stance)[0]
        new_data = {
            'pel_x': frame.p_pel_in_wf[0,0],
            'pel_y': frame.p_pel_in_wf[1,0],
            'pel_z': frame.p_pel_in_wf[2,0],

            'lf_x': frame.p_lf_in_wf[0,0],
            'lf_y': frame.p_lf_in_wf[1,0],
            'lf_z': frame.p_lf_in_wf[2,0],

            'rf_x': frame.p_rf_in_wf[0,0],
            'rf_y': frame.p_rf_in_wf[1,0],
            'rf_z': frame.p_rf_in_wf[2,0],

            'x': var['x'][0,0],
            'y': var['y'][0,0],
            
            'Ly': var['x'][1, 0],
            'Lx': var['y'][1, 0],
        }
        # 建立一筆資料的 DataFrame
        new_df = pd.DataFrame([new_data])
        # 使用 pd.concat 進行疊加
        mea_store = pd.concat([mea_store, new_df], ignore_index=True)
        return mea_store
        
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
