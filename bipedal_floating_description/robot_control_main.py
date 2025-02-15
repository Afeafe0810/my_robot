import rclpy; from rclpy.node import Node
from std_msgs.msg import Float64MultiArray 

import numpy as np; np.set_printoptions(precision=2)


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
 
    def main_controller_callback(self):
        #==========拿取訂閱值==========#
        p_base_in_wf, r_base_to_wf, state, _, _, jp, jv = self.ros.returnSubData()
        
        #==========更新可視化的機器人==========#
        config = self.robot.update_VizAndMesh(jp)
        
        #==========更新frame==========#
        self.frame.updateFrame(self.robot, config, p_base_in_wf, r_base_to_wf, jp)        
        
        # #========接觸判斷========#
        # contact = {
        #     'lf': (self.frame.p_lf_in_wf[2,0] <= 0.01),
        #     'rf': (self.frame.p_lf_in_wf[2,0] <= 0.01)
        # }
        # #TODO contact移到torque內
        # contact_lf, contact_rf = contact['lf'], contact['rf']

        #========支撐狀態切換=====#
        self._setStance(state)
        
        #========軌跡規劃========#
        ref = self.traj.plan(state, self.frame, self.stance)

        #========扭矩控制========#
        torque = self.ctrl.update_torque(self.frame, self.robot, ref, state, self.stance, self.stance_past, jp, jv)
        self.ros.publisher['effort'].publish( Float64MultiArray(data = torque) )
        
        self.stance_past = self.stance

    def _setStance(self, state):
        """掌管state 0, 1, 2的支撐腳邏輯"""
        if state in [0, 1, 2]:
            #(這邊不能用return寫，否則state30會是None)
            self.stance = ['lf', 'rf']
            
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
