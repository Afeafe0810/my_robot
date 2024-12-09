#================ import library ========================#
import rclpy
from rclpy.node import Node

import numpy as np; np.set_printoptions(precision=2)
#================ import other code =====================#
from utils.robot_control_init import ULC_init
#========================================================#


class UpperLevelController(Node):
    def __init__(self):
        super().__init__('upper_level_controllers')
        
        ULC_init_instance = ULC_init()
        self.callbackCount = 0 #每5次run一次 main_callback,做decimate(down sampling)降低振盪
        self.publishers = ULC_init.create_publishers(self)
        self.subscribers = ULC_init.create_subscribers(self)
        
        
        
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
