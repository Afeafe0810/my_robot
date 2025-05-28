import numpy as np; np.set_printoptions(precision=2)
from numpy.typing import NDArray
from rclpy.node import Node
from typing import Callable
from copy import deepcopy
from scipy.spatial.transform import Rotation as R

from std_msgs.msg import Float64MultiArray 
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ContactsState
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import WrenchStamped

#================ import other code =====================#
from utils.config import Config
from utils.signal_process import Dsp
#========================================================#


class ROSInterfaces:
    """
    負責處理ROS節點的訂閱與發佈, 設做全局的class
    
    - 使用方法:
        - 需要先init一次, 在各個模組內只要import就好
    
    - 屬性:
        - publishers
        - subscribers
    """

    @classmethod
    def init(cls, node: Node, main_callback: Callable ):
        cls.publishers = MyPublishers(node)
        cls.subscribers = MySubscribers(node, main_callback)

class MyPublishers:
    def __init__(self, node: Node):
        # 控制命令
        self.effort = MyPublisher(node, '/effort_controllers/commands')
        
        #只是用來追蹤數據
        # self.position = MyPublisher(node, '/position_controller/commands')
        # self.velocity = MyPublisher(node, '/velocity_controller/commands')
        # self.vcmd = MyPublisher(node, '/velocity_command/commands')
        # self.gravity_l = MyPublisher(node, '/l_gravity')
        # self.gravity_r = MyPublisher(node, '/r_gravity')
        # self.alip_x = MyPublisher(node, '/alip_x_data')
        # self.alip_y = MyPublisher(node, '/alip_y_data')
        # self.torque_l = MyPublisher(node, '/torqueL_data')
        # self.torque_r = MyPublisher(node, '/torqueR_data')
        # self.ref = MyPublisher(node, '/ref_data')
        # self.pel = MyPublisher(node, '/px_data')
        # self.com = MyPublisher(node, '/com_data')
        # self.lf = MyPublisher(node, '/lx_data')
        # self.rf = MyPublisher(node, '/rx_data')
        
        # self.joint_trajectory_controller = MyPublisher(node, '/joint_trajectory_controller/joint_trajectory', JointTrajectory)

class MyPublisher:
    """發佈器
    
    - method:
        - publish(msg): 目前只支援 Float64MultiArray 直接 publish
    """
    def __init__(self, node: Node, topic: str, msg_type: type = Float64MultiArray, qos_profile: int = 10):
        self._msg_type = msg_type
        self._publisher = node.create_publisher(msg_type, topic, qos_profile)
        
    def publish(self, msg):
        """目前只支援 Float64MultiArray 直接 publish"""
        if self._msg_type == Float64MultiArray:
            self._publisher.publish(self._msg_type(data = msg))
        else:
            raise NotImplementedError

class MySubscribers:
    """ 訂閱了base, state, contact, ft force, joint angle/velocity """
    def __init__(self, node: Node, main_callback: Callable):
        self.base = BaseSubscriber(node)
        self.state = StateSubsciber(node)
        self.lf_contact = ContactSubscriber(node, '/l_foot/bumper_demo')
        self.rf_contact = ContactSubscriber(node, '/r_foot/bumper_demo')
        self.lf_force = ForceSubscriber(node, '/lf_sensor/wrench')
        self.rf_force = ForceSubscriber(node, '/rf_sensor/wrench')
        self.jp = JointSubsciber(node, main_callback)

    def return_data(self) -> list[NDArray, NDArray, float, dict[str, bool], NDArray, NDArray, dict[str, float], dict[str, NDArray]]:
        #微分得到速度(飽和)，並濾波
        jp = Dsp.FILTER_JP.filt(self.jp.jp)
        _jv = np.clip( Dsp.DIFFTER_JP.diff(jp), -0.75, 0.75)
        jv = Dsp.FILTER_JV.filt(_jv)
        
        is_contact_ft = {'lf' : self.lf_contact.is_contact, 'rf' : self.rf_contact.is_contact}
        force_ft = {'lf' : self.lf_force.force[2,0], 'rf' : self.rf_force.force[2,0]}
        tau_ft = {'lf' : self.lf_force.tau, 'rf' : self.rf_force.tau}
        
        return list( map( deepcopy,
            [ 
                self.base.p_base_in_wf,
                self.base.r_base_to_wf,
                self.state.state,
                is_contact_ft,
                jp, 
                jv,
                force_ft, 
                tau_ft
            ]
        ))

class _AbstractSubscriber:
    def __init__(self, node: Node, msg_type: type, topic: str):
        self._subsciber = node.create_subscription(msg_type, topic, self._callback, 10)
        
    def _callback(self, msg):
        raise NotImplementedError
        
class BaseSubscriber(_AbstractSubscriber):
    """來自 bipedal_floation.xacro 的<libgazebo_ros_p3d.so>"""
    
    def __init__(self, node: Node):
        self.p_base_in_wf = None
        self.r_base_to_wf = None
        super().__init__(node, Odometry, '/odom')
                                                    
    def _callback(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation #四元數法
        self.p_base_in_wf = np.vstack(( p.x, p.y, p.z ))
        self.r_base_to_wf = R.from_quat(( q.x, q.y, q.z, q.w )).as_matrix()
        
class StateSubsciber(_AbstractSubscriber):
    '''state是我們控制策略, 用pub與subscribe來控制, 來自於我們手動pub'''
    
    def __init__(self, node: Node):
        
        self.state = 0.0
        super().__init__(node, Float64MultiArray, 'state_topic')
        
    def _callback(self, msg: Float64MultiArray):
        self.state = msg.data[0]

class ContactSubscriber(_AbstractSubscriber):
    '''可以判斷是否『接觸』, 無法判斷是否『踩穩』, 來自bipedal_floating.gazebo的 <libgazebo_ros_bumper.so>'''
    
    def __init__(self, node: Node, topic: str):
        self.is_contact = True
        super().__init__(node, ContactsState, topic)
        
    def _callback(self, msg: ContactsState):
        self.is_contact = bool(msg.states)

class ForceSubscriber(_AbstractSubscriber):
    """來自bipedal_floation.xacro的插件 <libgazebo_ros_ft_sensor.so>"""
    
    def __init__(self, node: Node, topic: str):
        self.force = None
        self.tau = None
        super().__init__(node, WrenchStamped, topic)
        
    def _callback(self, msg: WrenchStamped):
        force = msg.wrench.force
        torque = msg.wrench.torque
        
        self.force = np.vstack(( force.x, force.y, force.z ))
        self.tau = np.vstack(( torque.x, torque.y, torque.z ))

class JointSubsciber(_AbstractSubscriber):
    """來自bipedal_floation.xacro的插件 <libgazebo_ros2_control.so>, effort_controller.yaml"""
    
    def __init__(self, node: Node, main_callback: Callable):
        self.main_callback = main_callback #引入main_callback來持續呼叫
        self.callback_count = 0 #每5次會呼叫一次maincallback
        self.jp = None
        super().__init__(node, JointState, '/joint_states')
        
    def _callback(self, msg: JointState):
        '''訂閱jp, callback主程式'''
        if len(msg.name) == 12:
            # JointState不會按照順序訂閱也無法改，一定要比對 msg.name
            jp_pair = {jnt: value for jnt, value in zip( msg.name, msg.position) }
            self.jp = np.vstack([ jp_pair[jnt] for jnt in Config.JNT_ORDER_LITERAL ])

        self.callback_count += 1
        if self.callback_count == 5:
            self.callback_count = 0 
            self.main_callback()
