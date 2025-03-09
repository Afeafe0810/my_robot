import numpy as np; np.set_printoptions(precision=2)
from rclpy.node import Node
from typing import Callable
import pinocchio as pin #2.6.21
import pink #2.1.0
from sys import argv
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
    負責處理ROS節點的訂閱與發佈
    - 屬性:
        - publisher (dict)
        - subscriber (dict)
        
    - 方法:
        - returnSubData: 回傳訂閱器的數據，並進行速度微分和濾波, 包含:
            - base的位置與旋轉矩陣
            - 機器人控制模式state
            - 左右腳是否有接觸地面 (未必有踩穩)
            - 關節角度 (同時也每五次呼叫主程式的main_callback)
    """

    def __init__(self, node: Node, main_callback: Callable ):
        
        #=========初始化===========#
        self._p_base_in_wf = self._r_base_to_wf = self._jp = None
        self._state = 0.0
        self._is_contact_lf = self._is_contact_rf = True
        self._force_lf = self._tau_lf = None
        self._force_rf = self._tau_rf = None

        
        self._callback_count = 0 #每5次會呼叫一次maincallback
        self._main_callback = main_callback #引入main_callback來持續呼叫
        
        #=========ROS的訂閱與發布===========#
        
        self.publisher = self._createPublishers(node)
        self.subscriber = self._createSubscribers(node)
    
    #=========對外主要接口===========#
    def returnSubData(self) -> tuple[np.ndarray, np.ndarray, float, dict[str, bool], np.ndarray, np.ndarray, dict[str, float], dict[str, float]]:
        '''回傳訂閱器的data'''
        
        #微分得到速度(飽和)，並濾波
        _jv = np.clip( Dsp.DIFFTER_JP.diff(self._jp), -0.75, 0.75)
        jv = Dsp.FILTER_JV.filt(_jv) #TODO 原本舊的沒有濾波, 甚至設成0
        
        is_contact = {'lf' : self._is_contact_lf, 'rf' : self._is_contact_rf}
        force_ft = {'lf' : self._force_lf[2,0], 'rf' : self._force_rf[2,0]}
        tau_ft = {'lf' : self._tau_lf, 'rf' : self._tau_rf}
        
        return list( map( deepcopy,
            [ 
                self._p_base_in_wf,
                self._r_base_to_wf,
                self._state,
                is_contact,
                self._jp, 
                jv, 
                force_ft, 
                tau_ft
            ]
        ))
    
    #=========發布器, 訂閱器建立===========#
    @staticmethod
    def _createPublishers(node: Node):
        '''
        建立發布器, 其中effort publisher是ROS2-control的力矩, 負責控制各個關節的力矩
            ->我們所有程式目的就是為了pub關節扭矩
        '''
        return {
            #只有effort才是真正的控制命令，其他只是用來追蹤數據
            "effort"    : node.create_publisher( Float64MultiArray , '/effort_controllers/commands',  10),
            
            "position"  : node.create_publisher( Float64MultiArray , '/position_controller/commands', 10),
            "velocity"  : node.create_publisher( Float64MultiArray , '/velocity_controller/commands', 10),
            "vcmd"      : node.create_publisher( Float64MultiArray , '/velocity_command/commands',    10),
            "gravity_l" : node.create_publisher( Float64MultiArray , '/l_gravity',                    10),
            "gravity_r" : node.create_publisher( Float64MultiArray , '/r_gravity',                    10),
            "alip_x"    : node.create_publisher( Float64MultiArray , '/alip_x_data',                  10),
            "alip_y"    : node.create_publisher( Float64MultiArray , '/alip_y_data',                  10),
            "torque_l"  : node.create_publisher( Float64MultiArray , '/torqueL_data',                 10),
            "torque_r"  : node.create_publisher( Float64MultiArray , '/torqueR_data',                 10),
            "ref"       : node.create_publisher( Float64MultiArray , '/ref_data',                     10),
            "joint_trajectory_controller" : node.create_publisher(
                JointTrajectory , '/joint_trajectory_controller/joint_trajectory', 10
            ),
            "pel"       : node.create_publisher( Float64MultiArray , '/px_data',                      10),
            "com"       : node.create_publisher( Float64MultiArray , '/com_data',                     10),
            "lf"        : node.create_publisher( Float64MultiArray , '/lx_data',                      10),
            "rf"        : node.create_publisher( Float64MultiArray , '/rx_data',                      10),
        }

    def _createSubscribers(self, node: Node):
        '''主要是為了訂閱base, joint_states, state'''
        return{
            "base"         : node.create_subscription( Odometry,          '/odom',               self._update_baseInWf_callback, 10 ),
            "state"        : node.create_subscription( Float64MultiArray, 'state_topic',         self._update_state_callback,    10 ),
            "lf_contact"   : node.create_subscription( ContactsState,     '/l_foot/bumper_demo', self._update_contact_callback,  10 ),
            "rf_contact"   : node.create_subscription( ContactsState,     '/r_foot/bumper_demo', self._update_contact_callback,  10 ),
            "joint_states" : node.create_subscription( JointState,        '/joint_states',       self._update_jp_callback,       10 ),
            "lf_force"     : node.create_subscription( WrenchStamped,     '/lf_sensor/wrench',   self._update_lfForce_callback,  10 ),
            "rf_force"     : node.create_subscription( WrenchStamped,     '/rf_sensor/wrench',   self._update_rfForce_callback,  10 ),
        }

    #=========訂閱Callback===========#
    def _update_baseInWf_callback(self, msg:Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation #四元數法
        self._p_base_in_wf = np.vstack(( p.x, p.y, p.z ))
        self._r_base_to_wf = R.from_quat(( q.x, q.y, q.z, q.w )).as_matrix()
    
    def _update_state_callback(self, msg:Float64MultiArray):
        '''state是我們控制的模式, 用pub與subscibe來控制'''
        self._state = msg.data[0]
  
    def _update_contact_callback(self, msg:ContactsState ):
        '''可以判斷是否『接觸』, 無法判斷是否『踩穩』'''
        if msg.header.frame_id == 'l_foot_1':
            self._is_contact_lf = len(msg.states)>=1
        elif msg.header.frame_id == 'r_foot_1':
            self._is_contact_rf = len(msg.states)>=1
            
    def _update_jp_callback(self, msg:JointState ):
        '''訂閱jp, 且每5次會callback主程式一次'''
        if len(msg.position) == 12:
            jp_pair = {jnt: value for jnt,value in zip( Config.JNT_ORDER_SUB, msg.position) }
            self._jp = np.vstack([ jp_pair[jnt] for jnt in Config.JNT_ORDER_LITERAL ])

        self._callback_count += 1
        if self._callback_count == 5:
            self._callback_count = 0
            self._main_callback()
    
    def _update_lfForce_callback(self, msg: WrenchStamped):
        force = msg.wrench.force
        torque = msg.wrench.torque
        
        self._force_lf = np.vstack(( force.x, force.y, force.z ))
        self._tau_lf = np.vstack(( torque.x, torque.y, torque.z ))

    def _update_rfForce_callback(self, msg: WrenchStamped):
        """ 處理右腳感測器數據 """
        force = msg.wrench.force
        torque = msg.wrench.torque
        
        self._force_rf = np.vstack(( force.x, force.y, force.z ))
        self._tau_rf = np.vstack(( torque.x, torque.y, torque.z ))
    
class RobotModel:
    """
    一個使用 Pinocchio 和 Meshcat 進行可視化的機器人模型處理類別。
    - 屬性:
        - bipedal_floating: 從骨盆建下來的模擬模型。
        - stance_l: 從左腳掌往上建的左單腳。
        - stance_r: 從右腳掌往上建的右單腳。
        - bipedal_l: 從左腳掌建起的雙腳。
        - bipedal_r: 從右腳掌建起的雙腳。
    - 方法:
        - update_VizAndMesh: 給定關節轉角，更新機器人模型，回傳機器人的 configuration。
    """
    def __init__(self):
        #=========建立機器人模型===========#
        self._meshrobot = self._loadMeshcatModel("/bipedal_floating.pin.urdf") #Pinnocchio藍色的機器人

        self.bipedal_floating = SimpleModel("/bipedal_floating.xacro")  #從骨盆建下來的模擬模型
        self.stance_l         = SimpleModel("/stance_l.xacro")          #從左腳掌往上建的左單腳
        self.stance_r         = SimpleModel("/stance_r_gravity.xacro")  #從右腳掌往上建的右單腳
        self.bipedal_l        = SimpleModel("/bipedal_l_gravity.xacro") #從左腳掌建起的雙腳
        self.bipedal_r        = SimpleModel("/bipedal_r_gravity.xacro") #從右腳掌建起的雙腳
        
        #=========可視化msehcat===========#
        self._viz = self._meshcatVisualize(self._meshrobot)
        self.update_VizAndMesh(self._meshrobot.q0) #可視化模型的初始關節角度
        
    def update_VizAndMesh(self, jp: np.ndarray) -> pink.Configuration:
        '''給定關節轉角, 更新機器人模型, 回傳機器人的configuration'''
        config = pink.Configuration(self._meshrobot.model, self._meshrobot.data, jp)
        self._viz.display(config.q)
        return config
        
    @staticmethod
    def _loadMeshcatModel(urdf_path: str):
        '''高級動力學模型'''
        robot = pin.RobotWrapper.BuildFromURDF(
            filename = Config.ROBOT_MODEL_DIR + urdf_path,
            package_dirs = ["."],
            root_joint=None,
        )
        print(f"URDF description successfully loaded in {robot}")
        print(robot.model)
        print(robot.q0)
        return robot
    
    @staticmethod
    def  _meshcatVisualize(meshrobot: pin.RobotWrapper):
        '''可視化高級動力學模型'''
        viz = pin.visualize.MeshcatVisualizer( meshrobot.model, meshrobot.collision_model, meshrobot.visual_model )
        meshrobot.setVisualizer(viz, init=False)
        viz.initViewer(open=True)
        viz.loadViewerModel()

        return viz

class SimpleModel:
    '''基礎動力學模型, 用來算重力矩和質心位置'''
    def __init__(self, urdf_path: str):
        self.model, self.data = self._loadSimpleModel(urdf_path)
    
    @staticmethod
    def _loadSimpleModel(urdf_path: str):
        urdf_filename = Config.ROBOT_MODEL_DIR + urdf_path if len(argv)<2 else argv[1]
        model = pin.buildModelFromUrdf(urdf_filename)
        print(f'{model.name = }')
        model_data = model.createData()
        
        return model, model_data