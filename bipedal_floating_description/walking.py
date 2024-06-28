import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray 

from sensor_msgs.msg import JointState

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import pinocchio as pin
import qpsolvers

import pink
from pink import solve_ik
from pink.tasks import FrameTask, JointCouplingTask, PostureTask
import meshcat_shapes
import qpsolvers



import numpy as np
np.set_printoptions(precision=2)

from sys import argv
from os.path import dirname, join, abspath
import os
import copy
import math
from scipy.spatial.transform import Rotation as R



class UpperLevelController(Node):

    def __init__(self):
        super().__init__('upper_level_controllers')

        self.velocity_publisher = self.create_publisher(Float64MultiArray , '/velocity_controller/commands', 10)
        #init variables as self
        self.jp_sub = np.zeros(12)
        self.jv_sub = np.zeros(12)

        # self.Q0 = np.array([0.0, 0.0, -0.37, 0.74, -0.36, 0.0, 0.0, 0.0, -0.37, 0.74, -0.36, 0.0])
        # self.Q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -0.37, 0.74, -0.36, 0.0, 0.0, 0.0, -0.37, 0.74, -0.36, 0.0])

        self.joint_trajectory_controller = self.create_publisher(JointTrajectory , '/joint_trajectory_controller/joint_trajectory', 10)

        self.joint_states_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10)
        self.joint_states_subscriber  # prevent unused variable warning


        self.robot = self.load_URDF("/home/ldsc/ros2_ws/src/bipedal_floating_description/urdf/bipedal_floating.pin.urdf")
        
        # Initialize meschcat visualizer
        self.viz = pin.visualize.MeshcatVisualizer(
            self.robot.model, self.robot.collision_model, self.robot.visual_model
        )
        self.robot.setVisualizer(self.viz, init=False)
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel()


        # Set initial robot configuration
        print(self.robot.model)
        print(self.robot.q0)
        self.init_configuration = pink.Configuration(self.robot.model, self.robot.data, self.robot.q0)
        self.viz.display(self.init_configuration.q)

        # Tasks initialization for IK
        self.tasks = self.tasks_init()

        self.timer_period = 0.01 # seconds
        self.timer = self.create_timer(self.timer_period, self.main_controller_callback)

        
    def load_URDF(self, urdf_path):
        robot = pin.RobotWrapper.BuildFromURDF(
                        filename=urdf_path,
                        package_dirs=["."],
                        # root_joint=pin.JointModelFreeFlyer(),
                        root_joint=None,
                        )
        
        print(f"URDF description successfully loaded in {robot}")
        return robot

    def tasks_init(self):
        # Tasks initialization for IK
        left_foot_task = FrameTask(
            "l_foot",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        pelvis_task = FrameTask(
            "base_link",
            position_cost=1.0,
            orientation_cost=0.0,
        )
        right_foot_task = FrameTask(
            "r_foot_1",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        posture_task = PostureTask(
            cost=1e-1,  # [cost] / [rad]
        )
        tasks = {
            # 'left_foot_task': left_foot_task,
            'pelvis_task': pelvis_task,
            # 'right_foot_task': right_foot_task,
            'posture_task': posture_task,
        }
        return tasks

    def collect_joint_data(self):
        joint_position = copy.deepcopy(self.jp_sub)
        joint_velocity = copy.deepcopy(self.jv_sub)

        joint_position = np.reshape(joint_position,(12,1))
        joint_velocity = np.reshape(joint_velocity,(12,1))
        return joint_position,joint_velocity

    def joint_states_callback(self, msg):
        
        # Original ndarray order
        original_order = [
            'L_Hip_Yaw', 'L_Hip_Pitch', 'L_Knee_Pitch', 'L_Ankle_Pitch', 
            'L_Ankle_Roll', 'R_Hip_Roll', 'R_Hip_Yaw', 'R_Knee_Pitch', 
            'R_Hip_Pitch', 'R_Ankle_Pitch', 'L_Hip_Roll', 'R_Ankle_Roll'
        ]

        # Desired order
        desired_order = [
            'L_Hip_Roll', 'L_Hip_Yaw', 'L_Hip_Pitch', 'L_Knee_Pitch', 
            'L_Ankle_Pitch', 'L_Ankle_Roll', 'R_Hip_Roll', 'R_Hip_Yaw', 
            'R_Hip_Pitch', 'R_Knee_Pitch', 'R_Ankle_Pitch', 'R_Ankle_Roll'
        ]

        if len(msg.velocity) == 12:
            velocity_order_dict = {joint: value for joint, value in zip(original_order, np.array(msg.velocity))}

            self.jv_sub = np.array([velocity_order_dict[joint] for joint in desired_order])

        if len(msg.position) == 12:
            position_order_dict = {joint: value for joint, value in zip(original_order, np.array(msg.position))}
            self.jp_sub = np.array([position_order_dict[joint] for joint in desired_order])

        self.collect_joint_data()


        # print(msg.position)
        # print(msg.velocity)

        # print("-------")

    def xyz_rotation(self,axis,theta):
        cos = math.cos
        sin = math.sin
        R = np.array((3,3))
        if axis == 'x':
            R = np.array([[1,0,0],[0,cos(theta),-sin(theta)],[0,sin(theta),cos(theta)]])
        elif axis == 'y':
            R = np.array([[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]])
        elif axis == 'z':
            R = np.array([[cos(theta),-sin(theta),0],[sin(theta),cos(theta),0],[0,0,1]])
        return R    

    def rotation_matrix(self,joint_position):
        jp = copy.deepcopy(joint_position)

        #骨盆姿態(要確認！)
        self.RP = np.array([[1,0,0],[0,1,0],[0,0,1]])
        # 各關節角度
        Theta1 = jp[0,0] #L_Hip_Roll
        Theta2 = jp[1,0]
        Theta3 = jp[2,0]
        Theta4 = jp[3,0]
        Theta5 = jp[4,0]
        Theta6 = jp[5,0] #L_Ankle_Roll

        Theta7 = jp[6,0] #R_Hip_Roll
        Theta8 = jp[7,0]
        Theta9 = jp[8,0]
        Theta10 = jp[9,0]
        Theta11 = jp[10,0]
        Theta12 = jp[11,0] #R_Ankle_Roll

        #calculate rotation matrix
        self.L_R01 = self.xyz_rotation('x',Theta1) #L_Hip_roll
        self.L_R12 = self.xyz_rotation('z',Theta2)
        self.L_R23 = self.xyz_rotation('y',Theta3)
        self.L_R34 = self.xyz_rotation('y',Theta4)
        self.L_R45 = self.xyz_rotation('y',Theta5)
        self.L_R56 = self.xyz_rotation('x',Theta6) #L_Ankle_roll

        self.R_R01 = self.xyz_rotation('x',Theta7) #R_Hip_roll
        self.R_R12 = self.xyz_rotation('z',Theta8)
        self.R_R23 = self.xyz_rotation('y',Theta9)
        self.R_R34 = self.xyz_rotation('y',Theta10)
        self.R_R45 = self.xyz_rotation('y',Theta11)
        self.R_R56 = self.xyz_rotation('x',Theta12) #R_Ankle_roll

    def relative_axis(self):
        self.AL1 = self.RP@(np.array([[1],[0],[0]])) #L_Hip_roll
        self.AL2 = self.RP@self.L_R01@(np.array([[0],[0],[1]])) 
        self.AL3 = self.RP@self.L_R01@self.L_R12@(np.array([[0],[1],[0]])) 
        self.AL4 = self.RP@self.L_R01@self.L_R12@self.L_R23@(np.array([[0],[1],[0]]))
        self.AL5 = self.RP@self.L_R01@self.L_R12@self.L_R23@self.L_R34@(np.array([[0],[1],[0]])) 
        self.AL6 = self.RP@self.L_R01@self.L_R12@self.L_R23@self.L_R34@self.L_R45@(np.array([[1],[0],[0]])) #L_Ankle_Roll
        # print("AL1: ",self.AL1)
        # print("AL2: ",self.AL2)
        # print("AL3: ",self.AL3)
        # print("AL4: ",self.AL4)
        # print("AL5: ",self.AL5)
        # print("AL6: ",self.AL6)  

        self.AR1 = self.RP@(np.array([[1],[0],[0]])) #R_Hip_roll
        self.AR2 = self.RP@self.R_R01@(np.array([[0],[0],[1]])) 
        self.AR3 = self.RP@self.R_R01@self.R_R12@(np.array([[0],[1],[0]])) 
        self.AR4 = self.RP@self.R_R01@self.R_R12@self.R_R23@(np.array([[0],[1],[0]]))
        self.AR5 = self.RP@self.R_R01@self.R_R12@self.R_R23@self.R_R34@(np.array([[0],[1],[0]])) 
        self.AR6 = self.RP@self.R_R01@self.R_R12@self.R_R23@self.R_R34@self.R_R45@(np.array([[1],[0],[0]])) #R_Ankle_Roll
        # print("AR1: ",self.AR1)
        # print("AR2: ",self.AR2)
        # print("AR3: ",self.AR3)
        # print("AR4: ",self.AR4)
        # print("AR5: ",self.AR5)
        # print("AR6: ",self.AR6) 

    def get_position(self,configuration):
        self.pelvis = configuration.get_transform_frame_to_world("pelvis_link")
        # print("p",pelvis)
        self.l_hip_roll = configuration.get_transform_frame_to_world("l_hip_yaw_1")
        self.l_hip_yaw = configuration.get_transform_frame_to_world("l_hip_pitch_1")
        self.l_hip_pitch = configuration.get_transform_frame_to_world("l_thigh_1")
        self.l_knee_pitch = configuration.get_transform_frame_to_world("l_shank_1")
        self.l_ankle_pitch = configuration.get_transform_frame_to_world("l_ankle_1")
        self.l_ankle_roll = configuration.get_transform_frame_to_world("l_foot_1")
        self.l_foot = configuration.get_transform_frame_to_world("l_foot")
        # print("l_foot:",l_foot.translation)
        self.r_hip_roll = configuration.get_transform_frame_to_world("r_hip_yaw_1")
        self.r_hip_yaw = configuration.get_transform_frame_to_world("r_hip_pitch_1")
        self.r_hip_pitch = configuration.get_transform_frame_to_world("r_thigh_1")
        self.r_knee_pitch = configuration.get_transform_frame_to_world("r_shank_1")
        self.r_ankle_pitch = configuration.get_transform_frame_to_world("r_ankle_1")
        self.r_ankle_roll = configuration.get_transform_frame_to_world("r_foot_1")
        self.r_foot = configuration.get_transform_frame_to_world("r_foot")
        # print("r_foot:",r_foot.translation)
          
    def get_posture(self):
        cos = math.cos
        sin = math.sin

        pelvis_p = np.reshape(copy.deepcopy(self.pelvis.translation),(3,1))
        l_foot_p = np.reshape(copy.deepcopy(self.l_foot.translation),(3,1))
        r_foot_p = np.reshape(copy.deepcopy(self.r_foot.translation),(3,1))

        pelvis_o = copy.deepcopy(self.pelvis.rotation)
        l_foot_o = copy.deepcopy(self.l_foot.rotation)
        r_foot_o = copy.deepcopy(self.r_foot.rotation)

        pR = R.from_matrix(pelvis_o).as_euler('zyx', degrees=False)   
        P_Yaw = pR[0]
        P_Pitch = pR[1]
        P_Roll = pR[2]

        lR = R.from_matrix(l_foot_o).as_euler('zyx', degrees=False) 
        L_Yaw = lR[0]
        L_Pitch = lR[1]
        L_Roll = lR[2]
        
        rR = R.from_matrix(r_foot_o).as_euler('zyx', degrees=False) 
        R_Yaw = rR[0]
        R_Pitch = rR[1]
        R_Roll = rR[2]

        self.PX = np.array([[pelvis_p[0,0]],[pelvis_p[1,0]],[pelvis_p[2,0]],[P_Roll],[P_Pitch],[P_Yaw]])
        self.LX = np.array([[l_foot_p[0,0]],[l_foot_p[1,0]],[l_foot_p[2,0]],[L_Roll],[L_Pitch],[L_Yaw]])
        self.RX = np.array([[r_foot_p[0,0]],[r_foot_p[1,0]],[r_foot_p[2,0]],[R_Roll],[R_Pitch],[R_Yaw]])

        self.L_Body_transfer = np.array([[cos(L_Pitch)*cos(L_Yaw), -sin(L_Yaw),0],
                                [cos(L_Pitch)*sin(L_Yaw), cos(L_Yaw), 0],
                                [-sin(L_Pitch), 0, 1]])  
        
        self.R_Body_transfer = np.array([[cos(R_Pitch)*cos(R_Yaw), -sin(R_Yaw),0],
                                [cos(R_Pitch)*sin(R_Yaw), cos(R_Yaw), 0],
                                [-sin(R_Pitch), 0, 1]])  
        
    def left_leg_jacobian(self):
        pelvis = np.reshape(copy.deepcopy(self.pelvis.translation),(3,1))
        l_hip_roll = np.reshape(copy.deepcopy(self.l_hip_roll.translation),(3,1))
        l_hip_yaw = np.reshape(copy.deepcopy(self.l_hip_yaw.translation),(3,1))
        l_hip_pitch = np.reshape(copy.deepcopy(self.l_hip_pitch.translation),(3,1))
        l_knee_pitch = np.reshape(copy.deepcopy(self.l_knee_pitch.translation),(3,1))
        l_ankle_pitch = np.reshape(copy.deepcopy(self.l_ankle_pitch.translation),(3,1))
        l_ankle_roll = np.reshape(copy.deepcopy(self.l_ankle_roll.translation),(3,1))
        l_foot = np.reshape(copy.deepcopy(self.l_foot.translation),(3,1))
        # print("l_hip_roll:",l_hip_roll)
        # print("l_hip_yaw:",l_hip_yaw)
        # print("l_hip_pitch:",l_hip_pitch)
        # print("l_knee_pitch:",l_knee_pitch)
        # print("l_ankle_pitch:",l_ankle_pitch)
        # print("l_ankle_roll:",l_ankle_roll)
        # print("l_foot:",l_foot)


        # print("2",l_knee_pitch,l_ankle_pitch,l_ankle_roll)
        # l_foot = copy.deepcopy(l_foot)
        JL1 = np.cross(self.AL1,(l_foot-l_hip_roll),axis=0)
        JL2 = np.cross(self.AL2,(l_foot-l_hip_yaw),axis=0)
        JL3 = np.cross(self.AL3,(l_foot-l_hip_pitch),axis=0)
        JL4 = np.cross(self.AL4,(l_foot-l_knee_pitch),axis=0)
        JL5 = np.cross(self.AL5,(l_foot-l_ankle_pitch),axis=0)
        JL6 = np.cross(self.AL6,(l_foot-l_ankle_roll),axis=0)

        JLL_upper = np.hstack((JL1, JL2,JL3,JL4,JL5,JL6))
        JLL_lower = np.hstack((self.AL1,self.AL2,self.AL3,self.AL4,self.AL5,self.AL6))    
        self.JLL = np.vstack((JLL_upper,JLL_lower))  
        # print(self.JLL)

        return self.JLL

    def right_leg_jacobian(self):
        pelvis = np.reshape(copy.deepcopy(self.pelvis.translation),(3,1))
        r_hip_roll = np.reshape(copy.deepcopy(self.r_hip_roll.translation),(3,1))
        r_hip_yaw = np.reshape(copy.deepcopy(self.r_hip_yaw.translation),(3,1))
        r_hip_pitch = np.reshape(copy.deepcopy(self.r_hip_pitch.translation),(3,1))
        r_knee_pitch = np.reshape(copy.deepcopy(self.r_knee_pitch.translation),(3,1))
        r_ankle_pitch = np.reshape(copy.deepcopy(self.r_ankle_pitch.translation),(3,1))
        r_ankle_roll = np.reshape(copy.deepcopy(self.r_ankle_roll.translation),(3,1))
        r_foot = np.reshape(copy.deepcopy(self.r_foot.translation),(3,1))
        # print("1:",l_hip_roll,l_hip_yaw,l_hip_pitch)
        # print("2",l_knee_pitch,l_ankle_pitch,l_ankle_roll)
        # l_foot = copy.deepcopy(l_foot)
        JR1 = np.cross(self.AR1,(r_foot-r_hip_roll),axis=0)
        JR2 = np.cross(self.AR2,(r_foot-r_hip_yaw),axis=0)
        JR3 = np.cross(self.AR3,(r_foot-r_hip_pitch),axis=0)
        JR4 = np.cross(self.AR4,(r_foot-r_knee_pitch),axis=0)
        JR5 = np.cross(self.AR5,(r_foot-r_ankle_pitch),axis=0)
        JR6 = np.cross(self.AR6,(r_foot-r_ankle_roll),axis=0)

        JRR_upper = np.hstack((JR1,JR2,JR3,JR4,JR5,JR6))
        JRR_lower = np.hstack((self.AR1,self.AR2,self.AR3,self.AR4,self.AR5,self.AR6))    
        self.JRR = np.vstack((JRR_upper,JRR_lower))  
        # print(self.JRR)
    
        return self.JRR

    def ref_cmd(self):
        #pelvis
        P_X_ref = 0.0
        P_Y_ref = 0.0
        P_Z_ref = 0.6
        P_Roll_ref = 0.0
        P_Pitch_ref = 0.0
        P_Yaw_ref = 0.0

        self.PX_ref = np.array([[P_X_ref],[P_Y_ref],[P_Z_ref],[P_Roll_ref],[P_Pitch_ref],[P_Yaw_ref]])

        #left_foot
        L_X_ref = 0.1
        L_Y_ref = 0.1
        L_Z_ref = 0.05
        L_Roll_ref = 0.0
        L_Pitch_ref = 0.0
        L_Yaw_ref = 0.0
        
        self.LX_ref = np.array([[L_X_ref],[L_Y_ref],[L_Z_ref],[L_Roll_ref],[L_Pitch_ref],[L_Yaw_ref]])

        #right_foot
        R_X_ref = 0.1
        R_Y_ref = -0.1
        R_Z_ref = 0.05
        R_Roll_ref = 0.0
        R_Pitch_ref = 0.0
        R_Yaw_ref = 0.0

        self.RX_ref = np.array([[R_X_ref],[R_Y_ref],[R_Z_ref],[R_Roll_ref],[R_Pitch_ref],[R_Yaw_ref]])

    def calculate_err(self):
        PX_ref = copy.deepcopy(self.PX_ref)
        LX_ref = copy.deepcopy(self.LX_ref)
        RX_ref = copy.deepcopy(self.RX_ref)
        PX = copy.deepcopy(self.PX)
        LX = copy.deepcopy(self.LX)
        RX = copy.deepcopy(self.RX)

        L_ref = LX_ref - PX_ref 
        R_ref = RX_ref - PX_ref
        L = LX - PX
        R = RX - PX 

        Le_dot = 100*(L_ref - L)
        Re_dot = 100*(R_ref - R)

        Lroll_error_dot = Le_dot[3,0]
        Lpitch_error_dot = Le_dot[4,0]
        Lyaw_error_dot = Le_dot[5,0]
        WL_x = self.L_Body_transfer[0,0]*Lroll_error_dot + self.L_Body_transfer[0,1]*Lpitch_error_dot
        WL_y = self.L_Body_transfer[1,0]*Lroll_error_dot + self.L_Body_transfer[1,1]*Lpitch_error_dot
        WL_z = self.L_Body_transfer[2,0]*Lroll_error_dot + self.L_Body_transfer[2,2]*Lyaw_error_dot

        Le_2 = np.array([[Le_dot[0,0]],[Le_dot[1,0]],[Le_dot[2,0]],[WL_x],[WL_y],[WL_z]])

        Rroll_error_dot = Re_dot[3,0]
        Rpitch_error_dot = Re_dot[4,0]
        Ryaw_error_dot = Re_dot[5,0]
        WR_x = self.R_Body_transfer[0,0]*Rroll_error_dot + self.R_Body_transfer[0,1]*Rpitch_error_dot
        WR_y = self.R_Body_transfer[1,0]*Rroll_error_dot + self.R_Body_transfer[1,1]*Rpitch_error_dot
        WR_z = self.R_Body_transfer[2,0]*Rroll_error_dot + self.R_Body_transfer[2,2]*Ryaw_error_dot

        Re_2 = np.array([[Re_dot[0,0]],[Re_dot[1,0]],[Re_dot[2,0]],[WR_x],[WR_y],[WR_z]])

        return Le_2,Re_2
    
    def velocity_cmd(self,Le_2,Re_2):

        L2 = copy.deepcopy(Le_2)
        R2 = copy.deepcopy(Re_2)

        # print(L2)

        Lw_d = np.dot(np.linalg.pinv(self.JLL),L2) 
        Rw_d = np.dot(np.linalg.pinv(self.JRR),R2) 

        # print(Lw_d)

        return Lw_d,Rw_d

    def main_controller_callback(self):
        joint_position,joint_velocity = self.collect_joint_data()
        self.rotation_matrix(joint_position)
        self.relative_axis()

        configuration = pink.Configuration(self.robot.model, self.robot.data,joint_position)
        self.get_position(configuration)
        self.get_posture()
        self.viz.display(configuration.q)
        JLL = self.left_leg_jacobian()
        JRR = self.right_leg_jacobian()
        self.ref_cmd()
        Le_2,Re_2 = self.calculate_err()
        VL,VR = self.velocity_cmd(Le_2,Re_2)

        # v = np.vstack((VL,VR))

        # #position
        # p = joint_position + self.timer_period*v

        # v = np.reshape(v,(12))
        # p = np.reshape(p,(12))
        # print(p)
        # self.velocity_publisher.publish(Float64MultiArray(data=v))

        v = np.zeros(12)
        p = np.array([0.0,0.0,-0.37,0.74,-0.36,0.0,0.0,0.0,-0.37,0.74,-0.36,0.0])



        trajectory_msg  = JointTrajectory()
        # trajectory_msg.header.stamp = self.get_clock().now().to_msg()
        trajectory_msg.header.frame_id= 'base_link'
        trajectory_msg.joint_names = [
            'L_Hip_Roll', 'L_Hip_Yaw', 'L_Hip_Pitch', 'L_Knee_Pitch', 
            'L_Ankle_Pitch', 'L_Ankle_Roll', 'R_Hip_Roll', 'R_Hip_Yaw', 
            'R_Hip_Pitch', 'R_Knee_Pitch', 'R_Ankle_Pitch', 'R_Ankle_Roll'
        ]
        point = JointTrajectoryPoint()
        point.positions = list(p)
        point.velocities = list(v)
        point.time_from_start = rclpy.duration.Duration(seconds=self.timer_period).to_msg()
        trajectory_msg.points.append(point)
        self.joint_trajectory_controller.publish(trajectory_msg)



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
