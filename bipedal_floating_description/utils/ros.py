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
    負責處理ROS節點的訂閱與發佈, 設做全局的class
    
    - 使用方法, 需要先init一次, 在各個模組內只要import就好
    
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

    @classmethod
    def init(cls, node: Node, main_callback: Callable ):
        
        #=========初始化===========#
        cls._p_base_in_wf = cls._r_base_to_wf = cls._jp = None
        cls._state = 0.0
        cls._is_contact_lf = cls._is_contact_rf = True
        cls._force_lf = cls._tau_lf = None
        cls._force_rf = cls._tau_rf = None

        
        cls._callback_count = 0 #每5次會呼叫一次maincallback
        cls._main_callback = main_callback #引入main_callback來持續呼叫
        
        #=========ROS的訂閱與發布===========#
        
        cls.publisher = cls._createPublishers(node)
        cls.subscriber = cls._createSubscribers(node)
    
    #=========對外主要接口===========#
    @classmethod
    def returnSubData(cls) -> tuple[np.ndarray, np.ndarray, float, dict[str, bool], np.ndarray, np.ndarray, dict[str, float], dict[str, float]]:
        '''回傳訂閱器的data'''
        
        #微分得到速度(飽和)，並濾波
        jp = Dsp.FILTER_JP.filt(cls._jp)
        _jv = np.clip( Dsp.DIFFTER_JP.diff(jp), -0.75, 0.75)
        jv = Dsp.FILTER_JV.filt(_jv)
        
        is_contact = {'lf' : cls._is_contact_lf, 'rf' : cls._is_contact_rf}
        force_ft = {'lf' : cls._force_lf[2,0], 'rf' : cls._force_rf[2,0]}
        tau_ft = {'lf' : cls._tau_lf, 'rf' : cls._tau_rf}
        
        return list( map( deepcopy,
            [ 
                cls._p_base_in_wf,
                cls._r_base_to_wf,
                cls._state,
                is_contact,
                jp, 
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

    @classmethod
    def _createSubscribers(cls, node: Node):
        '''主要是為了訂閱base, joint_states, state
        
        main_callback是和joint_states同步
        '''
        return{
            "base"         : node.create_subscription( Odometry,          '/odom',               cls._update_baseInWf_callback,  10 ),
            "state"        : node.create_subscription( Float64MultiArray, 'state_topic',         cls._update_state_callback,     10 ),
            "lf_contact"   : node.create_subscription( ContactsState,     '/l_foot/bumper_demo', cls._update_contact_callback,   10 ),
            "rf_contact"   : node.create_subscription( ContactsState,     '/r_foot/bumper_demo', cls._update_contact_callback,   10 ),
            "joint_states" : node.create_subscription( JointState,        '/joint_states',       cls._update_jpAndMain_callback, 10 ),
            "lf_force"     : node.create_subscription( WrenchStamped,     '/lf_sensor/wrench',   cls._update_lfForce_callback,   10 ),
            "rf_force"     : node.create_subscription( WrenchStamped,     '/rf_sensor/wrench',   cls._update_rfForce_callback,   10 ),
        }

    #=========訂閱Callback===========#
    @classmethod
    def _update_baseInWf_callback(cls, msg:Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation #四元數法
        cls._p_base_in_wf = np.vstack(( p.x, p.y, p.z ))
        cls._r_base_to_wf = R.from_quat(( q.x, q.y, q.z, q.w )).as_matrix()
    
    @classmethod
    def _update_state_callback(cls, msg:Float64MultiArray):
        '''state是我們控制的模式, 用pub與subscibe來控制'''
        cls._state = msg.data[0]
  
    @classmethod
    def _update_contact_callback(cls, msg:ContactsState ):
        '''可以判斷是否『接觸』, 無法判斷是否『踩穩』'''
        if msg.header.frame_id == 'l_foot_1':
            cls._is_contact_lf = len(msg.states)>=1
        elif msg.header.frame_id == 'r_foot_1':
            cls._is_contact_rf = len(msg.states)>=1
            
    @classmethod
    def _update_jpAndMain_callback(cls, msg:JointState ):
        '''訂閱jp, callback主程式'''
        if len(msg.position) == 12:
            jp_pair = {jnt: value for jnt,value in zip( Config.JNT_ORDER_SUB, msg.position) }
            cls._jp = np.vstack([ jp_pair[jnt] for jnt in Config.JNT_ORDER_LITERAL ])

        cls._callback_count += 1
        if cls._callback_count == 5:
            cls._callback_count = 0
            cls._main_callback()
    
    @classmethod
    def _update_lfForce_callback(cls, msg: WrenchStamped):
        force = msg.wrench.force
        torque = msg.wrench.torque
        
        cls._force_lf = np.vstack(( force.x, force.y, force.z ))
        cls._tau_lf = np.vstack(( torque.x, torque.y, torque.z ))

    @classmethod
    def _update_rfForce_callback(cls, msg: WrenchStamped):
        """ 處理右腳感測器數據 """
        force = msg.wrench.force
        torque = msg.wrench.torque
        
        cls._force_rf = np.vstack(( force.x, force.y, force.z ))
        cls._tau_rf = np.vstack(( torque.x, torque.y, torque.z ))
    
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

        self.bipedal_from_pel = BipedalFromPel("/bipedal_floating.xacro") #從骨盆建下來的模擬模型
        self.bipedal_from_lf = BipedalFromLF("/bipedal_l_gravity.xacro") #從左腳掌建起的雙腳
        self.bipedal_from_rf = BipedalFromRF("/bipedal_r_gravity.xacro") #從右腳掌建起的雙腳
        self.single_lf = SingleLeftLeg("/stance_l.xacro") #從左腳掌往上建的左單腳
        self.single_rf = SingleRightLeg("/stance_r_gravity.xacro") #從右腳掌往上建的右單腳
        
        #=========可視化msehcat===========#
        self._viz = self._meshcatVisualize(self._meshrobot)
        self.update_VizAndMesh(self._meshrobot.q0) #可視化模型的初始關節角度
        
    def update_VizAndMesh(self, jp: np.ndarray) -> pink.Configuration:
        '''給定關節轉角, 更新機器人模型, 回傳機器人的configuration'''
        config = pink.Configuration(self._meshrobot.model, self._meshrobot.data, jp)
        self._viz.display(config.q)
        return config
    
    def gravity(self, jp: np.ndarray, state: float, stance: list[str], pa_lfTOpel_in_pf: np.ndarray, pa_rfTOpel_in_pf: np.ndarray) -> np.ndarray:
        
        def weighted(x, x0, x1, g0, g1) -> np.ndarray:
            return g0 +(g1-g0)/(x1-x0)*(x-x0)
            
        #==========半邊單腳模型==========#
        g_from_both_single_ft = np.vstack((
            self.single_lf.gravity(jp),
            self.single_rf.gravity(jp)
        ))
        
        #==========腳底建起的模型==========#
        g_from_bipedal_ft = {
            'lf': self.bipedal_from_lf.gravity(jp),
            'rf': self.bipedal_from_rf.gravity(jp)
        }
        
        match state:
            case 0 | 1 | 2:
                y_ftTOpel = {'lf': abs(pa_lfTOpel_in_pf[1,0]), 'rf': abs(pa_rfTOpel_in_pf[1,0])}
                
                #雙腳平衡時, 用距離判斷重心腳
                cf = 'lf' if y_ftTOpel['lf'] <= y_ftTOpel['rf'] else 'rf'
                
                return weighted(y_ftTOpel[cf], *[0, 0.1], *[g_from_bipedal_ft[cf], g_from_both_single_ft])
                
            case 30:
                cf, sf = stance
                
                return 0.3 * g_from_both_single_ft + 0.75* g_from_bipedal_ft[cf]
        
        
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

class _SimpleModel:
    '''基礎動力學模型, 用來算重力矩和質心位置'''
    def __init__(self, urdf_path: str):
        urdf = Config.ROBOT_MODEL_DIR + urdf_path
        model = pin.buildModelFromUrdf(urdf)
        print(f'{model.name = }')
        
        #機器人模型
        self.model, self.data = model, model.createData()
        
        #關節順序的轉換
        self.permut: np.ndarray = self._joint_permutation()
        self.inv_permut: np.ndarray = self._joint_inverse_permutation()
        
    @staticmethod
    def _joint_permutation()-> np.ndarray:
        return np.array([[]])
    
    def _joint_inverse_permutation(self)-> np.ndarray:
        """「方陣」置換矩陣的反函數 = 轉置"""
        return self._joint_permutation().T
    
    def gravity(self, jp: np.ndarray) -> np.ndarray:
        return self.inv_permut @ np.vstack(pin.computeGeneralizedGravity(self.model, self.data, self.permut@jp))
    
class BipedalFromPel(_SimpleModel):
    def com(self, jp: np.ndarray) -> np.ndarray:
        # TODO 學長有用其他模型建立, 不知道會不會有差, 但我目前是覺得就算有差也不可能差多少啦
        pin.centerOfMass(self.model, self.data, jp)
        return self.data.com[0].reshape(3,1)

class BipedalFromLF(_SimpleModel):
    @staticmethod
    def _joint_permutation()-> np.ndarray:
        """關節順序轉成5-0、6-11, 反轉須反向"""
        P = _pure_permute()
        O = np.zeros((6,6))
        I = np.eye(6,6)
        return np.block([[P, O], [O, I]])
    
class BipedalFromRF(_SimpleModel):
    @staticmethod
    def _joint_permutation()-> np.ndarray:
        """關節順序轉成11-6、0-5, 反轉須反向"""
        P = _pure_permute()
        O = np.zeros((6,6))
        I = np.eye(6,6)
        return np.block([[O, P], [I, O]])

class SingleLeftLeg(_SimpleModel):
    @staticmethod
    def _joint_permutation()-> np.ndarray:
        """關節順序轉成5-0, 反轉須反向"""
        P = _pure_permute()
        O = np.zeros((6,6))
        return np.block([P, O])
    
    @staticmethod
    def _joint_inverse_permutation():
        return _pure_permute()
    
class SingleRightLeg(_SimpleModel):
    @staticmethod
    def _joint_permutation()-> np.ndarray:
        """關節順序轉成11-6, 反轉須反向"""
        P = _pure_permute()
        O = np.zeros((6,6))
        return np.block([O, P])
    
    @staticmethod
    def _joint_inverse_permutation():
        return _pure_permute()

def _pure_permute():
    """6*6的純置換矩陣, 反轉且反向"""
    permut = np.zeros((6,6))
    for i,j in [(0,5), (1,4), (2,3), (3,2), (4,1), (5,0)]:
        permut[i,j] = -1
    return permut