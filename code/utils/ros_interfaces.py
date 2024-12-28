import rclpy; from rclpy.node import Node
from typing import Callable


from std_msgs.msg import Float64MultiArray 
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ContactsState
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectory

import pinocchio as pin
from sys import argv

import pink
import meshcat_shapes

import numpy as np; np.set_printoptions(precision=2)
from copy import deepcopy
from scipy.spatial.transform import Rotation as R

#================ import other code =====================#
from utils.config import Config
#========================================================#


class ROSInterfaces:
    def __init__(self, node: Node, main_callback: Callable ):
        
        #=========初始化===========#
        self.__p_base_in_wf = self.__r_base_to_wf = self.__jp = None
        self.__state = 0
        self.__contact_l = self.__contact_r = True
        
        self.__callback_time = 0 #每5次會呼叫一次maincallback
        self.__main_callback = main_callback #引入main_callback來持續呼叫
        
        #=========ROS的訂閱與發布===========#
        
        self.publisher = self.__createPublishers(node)
        self.subscriber = self.__createSubscribers(node)
        
        #=========建立機器人模型===========#
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
       
    def getSubDate(self):
        return [
            deepcopy(data) for data in [
                self.__p_base_in_wf, self.__r_base_to_wf, self.__state, self.__contact_l, self.__contact_r, self.__jp
            ]
        ]
    
    def update_VizAndMesh(self, jp):
        config = pink.Configuration(self.__meshrobot.model, self.__meshrobot.data, jp)
        self.__viz.display(config.q)
        return config
 
    @staticmethod
    def __createPublishers(node: Node):
        '''effort publisher是ROS2-control的力矩, 負責控制各個關節的力矩->我們程式的目的就是為了pub他'''
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
            "base": node.create_subscription( Odometry, '/odom', self.__base_in_wf, 10 ),
            "state": node.create_subscription( Float64MultiArray, 'state_topic', self.__state_callback, 10 ),
            "lf_contact": node.create_subscription( ContactsState, '/l_foot/bumper_demo', self.__contact_callback, 10 ),
            "rf_contact": node.create_subscription( ContactsState, '/r_foot/bumper_demo', self.__contact_callback, 10 ),
            "joint_states": node.create_subscription( JointState, '/joint_states', self.__joint_states_callback, 0 ),
        }

    def __base_in_wf(self, msg:Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation #四元數法
        self.__p_base_in_wf = np.vstack(( p.x, p.y, p.z ))
        self.__r_base_to_wf = R.from_quat(( q.x, q.y, q.z, q.w )).as_matrix()
    
    def __state_callback(self, msg:Float64MultiArray):
        self.__state = msg.data[0]
  
    def __contact_callback(self, msg:ContactsState ):
        
        if msg.header.frame_id == 'l_foot_1':
            self.__contact_l = len(msg.states)>=1
        elif msg.header.frame_id == 'r_foot_1':
            self.__contact_r = len(msg.states)>=1
            
    def __joint_states_callback(self, msg:JointState ):
        if len(msg.position) == 12:
            jp_pair = {jnt: value for jnt,value in zip( Config.JNT_ORDER_SUB, msg.position) }
            self.__jp = np.vstack([ jp_pair[jnt] for jnt in Config.JNT_ORDER_LITERAL ])

        self.__callback_time += 1
        if self.__callback_time == 5:
            self.__callback_time = 0
            self.__main_callback()
            
    @staticmethod
    def __loadMeshcatModel(urdf_path):
        robot = pin.RobotWrapper.BuildFromURDF(
            filename = Config.PINOCCHIO_MODEL_DIR + urdf_path,
            package_dirs = ["."],
            root_joint=None,
        )
        print(f"URDF description successfully loaded in {robot}")
        print(robot.model)
        print(robot.q0)
        return robot
    
    @staticmethod
    def __loadSimpleModel(urdf_path):
        
        urdf_filename = Config.PINOCCHIO_MODEL_DIR + urdf_path if len(argv)<2 else argv[1]
        model  = pin.buildModelFromUrdf(urdf_filename)
        print('model name: ' + model.name)
        model_data = model.createData()
        
        return model, model_data
     
    @staticmethod
    def  __meshcatVisualize(meshrobot):
        viz = pin.visualize.MeshcatVisualizer( meshrobot.model, meshrobot.collision_model, meshrobot.visual_model )
        meshrobot.setVisualizer(viz, init=False)
        viz.initViewer(open=True)
        viz.loadViewerModel()

        return viz
