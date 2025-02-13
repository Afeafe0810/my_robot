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

#================ import other code =====================#
from utils.config import Config
from utils.signal_process import Dsp
#========================================================#


class ROSInterfaces:
    """
    負責處理所有模擬相關的設置，包含
    1. ROS節點的訂閱與發佈: publisher、subscriber
    2. 訂閱的data: 利用 updateSubData 的method獲取
        \t- 包含base的位置與旋轉矩陣
        \t- 機器人控制模式state
        \t- 左右腳是否有接觸地面 (未必有踩穩)
        \t- 關節角度 (同時也每五次呼叫主程式的main_callback)
    3. meshcat模型的更新: 利用update_VizAndMesh, 且會回傳機器人的Configuration
    4. 不同重力矩模型的建立
    """

    def __init__(self, node: Node, main_callback: Callable ):
        
        #=========初始化===========#
        self.__p_base_in_wf = self.__r_base_to_wf = self.__jp = None
        self.__state = 0
        self.__contact_lf = self.__contact_rf = True
        
        self.__callback_count = 0 #每5次會呼叫一次maincallback
        self.__main_callback = main_callback #引入main_callback來持續呼叫
        
        #=========ROS的訂閱與發布===========#
        
        self.publisher = self.__createPublishers(node)
        self.subscriber = self.__createSubscribers(node)
        
        #=========建立機器人模型===========#
        #TODO: 最好再開一個class叫做robot
        self.__meshrobot = self.__loadMeshcatModel("/bipedal_floating.pin.urdf")
        self.meshrobot = self.__meshrobot
        self.bipedal_floating_model, self.bipedal_floating_data = self.__loadSimpleModel("/bipedal_floating.xacro") #從骨盆建下來的模擬模型
        self.stance_l_model,         self.stance_l_data         = self.__loadSimpleModel("/stance_l.xacro") #從左腳掌往上建的左單腳
        self.stance_r_model,         self.stance_r_data         = self.__loadSimpleModel("/stance_r_gravity.xacro") #從右腳掌往上建的右單腳
        self.bipedal_l_model,        self.bipedal_l_data        = self.__loadSimpleModel("/bipedal_l_gravity.xacro") #從左腳掌建起的雙腳
        self.bipedal_r_model,        self.bipedal_r_data        = self.__loadSimpleModel("/bipedal_r_gravity.xacro") #從右腳掌建起的雙腳
        
        #=========可視化msehcat===========#
        self.__viz = self.__meshcatVisualize(self.__meshrobot)
        self.update_VizAndMesh(self.__meshrobot.q0) #可視化模型的初始關節角度
        # Set initial robot configuration
       
    def updateSubData(self):
        '''回傳訂閱器的data'''
        #==========微分得到速度(飽和)，並濾波==========#
        __jv = np.clip( Dsp.DIFFTER["jp"].diff(self.__jp), -0.75, 0.75) #微分後設定飽和
        jv = Dsp.FILTER["jv"].filt(__jv)
        
        return list( map( deepcopy,
            [ self.__p_base_in_wf, self.__r_base_to_wf, self.__state, self.__contact_lf, self.__contact_rf, self.__jp, jv ]
        ))
    
    def update_VizAndMesh(self, jp: np.ndarray) -> pink.Configuration:
        '''給定關節轉角, 更新機器人模型, 回傳機器人的configuration'''
        config = pink.Configuration(self.__meshrobot.model, self.__meshrobot.data, jp)
        self.__viz.display(config.q)
        return config
 
    @staticmethod
    def __createPublishers(node: Node):
        '''
        建立發布器, 其中effort publisher是ROS2-control的力矩, 負責控制各個關節的力矩
        ->我們程式的目的就是為了pub他
        '''
        return {
            #只有effort才是真正的控制命令，其他只是用來追蹤數據
            "effort" : node.create_publisher(Float64MultiArray , '/effort_controllers/commands', 10),
            
            "position" : node.create_publisher(Float64MultiArray , '/position_controller/commands', 10),
            "velocity" : node.create_publisher(Float64MultiArray , '/velocity_controller/commands', 10),
            "vcmd" :  node.create_publisher(Float64MultiArray , '/velocity_command/commands', 10),
            "gravity_l" :  node.create_publisher(Float64MultiArray , '/l_gravity', 10),
            "gravity_r" : node.create_publisher(Float64MultiArray , '/r_gravity', 10),
            "alip_x" : node.create_publisher(Float64MultiArray , '/alip_x_data', 10),
            "alip_y" : node.create_publisher(Float64MultiArray , '/alip_y_data', 10),
            "torque_l" : node.create_publisher(Float64MultiArray , '/torqueL_data', 10),
            "torque_r" : node.create_publisher(Float64MultiArray , '/torqueR_data', 10),
            "ref" : node.create_publisher(Float64MultiArray , '/ref_data', 10),
            "joint_trajectory_controller" : node.create_publisher(
                JointTrajectory , '/joint_trajectory_controller/joint_trajectory', 10
            ),
            "pel" : node.create_publisher(Float64MultiArray , '/px_data', 10),
            "com" : node.create_publisher(Float64MultiArray , '/com_data', 10),
            "lf" : node.create_publisher(Float64MultiArray , '/lx_data', 10),
            "rf" : node.create_publisher(Float64MultiArray , '/rx_data', 10),
        }

    def __createSubscribers(self, node: Node):
        '''主要是為了訂閱base, joint_states, state'''
        return{
            "base": node.create_subscription( Odometry, '/odom', self.__update_baseInWf_callback, 10 ),
            "state": node.create_subscription( Float64MultiArray, 'state_topic', self.__update_state_callback, 10 ),
            "lf_contact": node.create_subscription( ContactsState, '/l_foot/bumper_demo', self.__update_contact_callback, 10 ),
            "rf_contact": node.create_subscription( ContactsState, '/r_foot/bumper_demo', self.__update_contact_callback, 10 ),
            "joint_states": node.create_subscription( JointState, '/joint_states', self.__update_jp_callback, 0 ),
        }

    def __update_baseInWf_callback(self, msg:Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation #四元數法
        self.__p_base_in_wf = np.vstack(( p.x, p.y, p.z ))
        self.__r_base_to_wf = R.from_quat(( q.x, q.y, q.z, q.w )).as_matrix()
    
    def __update_state_callback(self, msg:Float64MultiArray):
        '''state是我們控制的模式, 用pub與subscibe來控制'''
        self.__state = msg.data[0]
  
    def __update_contact_callback(self, msg:ContactsState ):
        '''可以判斷是否『接觸』, 無法判斷是否『踩穩』'''
        if msg.header.frame_id == 'l_foot_1':
            self.__contact_lf = len(msg.states)>=1
        elif msg.header.frame_id == 'r_foot_1':
            self.__contact_rf = len(msg.states)>=1
            
    def __update_jp_callback(self, msg:JointState ):
        '''訂閱jp, 且每5次會callback主程式一次'''
        if len(msg.position) == 12:
            jp_pair = {jnt: value for jnt,value in zip( Config.JNT_ORDER_SUB, msg.position) }
            self.__jp = np.vstack([ jp_pair[jnt] for jnt in Config.JNT_ORDER_LITERAL ])

        self.__callback_count += 1
        if self.__callback_count == 5:
            self.__callback_count = 0
            self.__main_callback()
            
    @staticmethod
    def __loadMeshcatModel(urdf_path: str):
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
    def __loadSimpleModel(urdf_path: str):
        '''基礎動力學模型, 只用來算重力矩'''
        urdf_filename = Config.ROBOT_MODEL_DIR + urdf_path if len(argv)<2 else argv[1]
        model  = pin.buildModelFromUrdf(urdf_filename)
        print('model name: ' + model.name)
        model_data = model.createData()
        
        return model, model_data
     
    @staticmethod
    def  __meshcatVisualize(meshrobot: pin.RobotWrapper):
        '''可視化高級動力學模型'''
        viz = pin.visualize.MeshcatVisualizer( meshrobot.model, meshrobot.collision_model, meshrobot.visual_model )
        meshrobot.setVisualizer(viz, init=False)
        viz.initViewer(open=True)
        viz.loadViewerModel()

        return viz
