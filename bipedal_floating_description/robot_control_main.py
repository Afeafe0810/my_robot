import rclpy; from rclpy.node import Node
from std_msgs.msg import Float64MultiArray 

import numpy as np; np.set_printoptions(precision=2)


#================ import other code =====================#
from utils.config import Config
from utils.ros_interfaces import ROSInterfaces
from utils.frame_kinermatic import RobotFrame
from utils.trajatory_planning import *
from utils.torque_control import *
#========================================================#

class UpperLevelController(Node):

    def __init__(self):
        #建立ROS的node
        super().__init__('upper_level_controllers')
        
        #負責模型與ROS的功能
        self.ros = ROSInterfaces(self, self.main_controller_callback)
        
        #負責量測各部位的位置與姿態
        self.frame = RobotFrame()
        
        #負責處理軌跡
        self.traj = Trajatory()
        
        #負責處理扭矩
        self.ctrl = TorqueControl()
        
        #============機器人的重要參數=====================#     
        
        #機器人的模式
        self.state = 0
        self.state_past = 0
        
        #主被動腳
        self.stance = ['lf', 'rf'] #第一個是支撐腳cf, 第二個是擺動腳sf
        self.stance_past = ['lf', 'rf'] #上個取樣時間的支撐腳與擺動腳
        
    def stance_change(self, state, alip_time):
        self.stance = ['lf', 'rf'] if state == 0 else\
                      ['lf', 'rf'] if state == 1 else\
                    self.stance #先不變

        # 時間到就做兩隻腳的切換
        if state == 30 and abs(alip_time-0.5) <= 0.005 :
            self.stance.reverse()
 
    def main_controller_callback(self):
        #==========拿取訂閱值==========#
        p_base_in_wf, r_base_to_wf, state, contact_lf, contact_rf, jp, jv = self.ros.updateSubData()
        
        #==========更新可視化的機器人==========#
        config = self.ros.update_VizAndMesh(jp)
        
        #==========更新frame==========#
        self.frame.updateFrame(self.ros, config, p_base_in_wf, r_base_to_wf, jp)

        px_in_lf,px_in_rf = self.frame.get_posture(self.frame.pa_pel_in_pf, self.frame.pa_lf_in_pf, self.frame.pa_rf_in_pf)
        
        
        #========接觸判斷========#
        contact_lf = (self.frame.p_lf_in_wf[2,0] <= 0.01)
        contact_rf = (self.frame.p_rf_in_wf[2,0] <= 0.01)

        #========支撐狀態切換=====#
        self.stance_change(state, self.traj.alip_time)
        cf, sf = self.stance
        cf_past, sf_past = self.stance_past
        
        #========軌跡規劃========#
        ref = self.traj.plan(state)

        #========扭矩控制========#
        torque = self.ctrl.update_torque(self.frame, jp, self.ros, self.stance, self.stance_past, px_in_lf, px_in_rf, contact_lf, contact_rf , state, ref, jv)
        self.ros.publisher['effort'].publish( Float64MultiArray(data = torque) )
        
        self.state_past, self.stance_past = self.state, self.stance

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
