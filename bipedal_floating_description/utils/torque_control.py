#================ import library ========================#
import numpy as np; np.set_printoptions(precision=2)
import copy
from scipy.spatial.transform import Rotation as R
import pinocchio as pin

#================ import library ========================#
from utils.ros_interfaces import ROSInterfaces
from utils.frame_kinermatic import RobotFrame
from utils.config import Config
from utils.signal_process import Dsp

class TorqueControl:
    
    @classmethod
    def update_torque(cls, frame: RobotFrame, jp, ros, cf_past, px_in_lf, px_in_rf, state, ref_pa_in_wf, jv, stance):
        if state == 0:
            torque = Innerloop.balance(jp, ros, stance, px_in_lf, px_in_rf)
        if state in [1, 2]:
            cf, sf = stance
            torque = cls.__kneecontrol(frame, ros, jp, px_in_lf, px_in_rf, ref_pa_in_wf, jv, stance, state)
            
            torque[sf][4:6] = cls.__swingAnkle_PDcontrol(sf, frame.r_lf_to_wf, frame.r_rf_to_wf)
            torque[cf][4:6] = cls.__alip_control(frame, cf, cf_past, frame.p_com_in_wf, frame.p_lf_in_wf, frame.p_rf_in_wf, ref_pa_in_wf)
            torque = np.vstack(( torque['lf'], torque['rf'] ))
        return torque
    
    @staticmethod
    def __kneecontrol(frame, ros, jp, px_in_lf, px_in_rf, ref_pa_in_wf, jv, stance, state):
        cf, _ = stance
        cmd_jv = Outterloop.get_jv_cmd(frame, ref_pa_in_wf, jv, stance, state)
        return Innerloop.innerloopDynamics(jv, cmd_jv, ros, jp, stance, px_in_lf, px_in_rf)
    
    @staticmethod
    def __swingAnkle_PDcontrol(sf, r_lf_to_wf, r_rf_to_wf):
        r_ft_to_wf = {
            'lf': r_lf_to_wf,
            'rf': r_rf_to_wf
        }
        _, *ayx_sf_in_wf = np.vstack(( R.from_matrix(r_ft_to_wf[sf]).as_euler('zyx', degrees=False) ))
        
        ref_jp = np.zeros((2,1))
        torque_ankle_sf = 0.1 * ( ref_jp - ayx_sf_in_wf )
        
        return torque_ankle_sf

    @staticmethod
    def __alip_control(frame:RobotFrame, cf, cf_past, p_com_in_wf, p_lf_in_wf, p_rf_in_wf, ref_pa_in_wf):
        ref_pa_com_in_wf, ref_pa_lf_in_wf, ref_pa_rf_in_wf = ref_pa_in_wf['pel'], ref_pa_in_wf['lf'], ref_pa_in_wf['rf']
        #質心相對L frame的位置
        p_ft_in_wf = {
            'lf': p_lf_in_wf,
            'rf': p_rf_in_wf
        }
        ref_pa_ft_in_wf = {
            'lf': ref_pa_lf_in_wf,
            'rf': ref_pa_rf_in_wf
        }
        x_cfTOcom_in_wf, y_cfTOcom_in_wf = ( p_com_in_wf - p_ft_in_wf[cf] ) [0:2,0]
        ref_x_cfTOcom_in_wf, ref_y_cfTOcom_in_wf = ( ref_pa_com_in_wf - ref_pa_ft_in_wf[cf] ) [0:2,0]

        #計算質心速度(v從世界座標下求出)
        vx_com_in_wf, vy_com_in_wf = Dsp.FILTER["v_com_in_wf"].filt(
            Dsp.DIFFTER["p_com_in_wf"].diff(p_com_in_wf) 
        ) [0:2,0]
        
        Ly_com_in_wf =  9 * vx_com_in_wf * 0.45
        Lx_com_in_wf = -9 * vy_com_in_wf * 0.45
        
        wx = np.vstack(( x_cfTOcom_in_wf, Ly_com_in_wf ))
        wy = np.vstack(( y_cfTOcom_in_wf, Lx_com_in_wf ))
        
        ref_wx = np.vstack(( ref_x_cfTOcom_in_wf, 0 ))
        ref_wy = np.vstack(( ref_y_cfTOcom_in_wf, 0 ))
        
        #xc & ly model(m=9 H=0.45 Ts=0.01)
        # Ax = np.array([[1,0.00247],[0.8832,1]])
        # Bx = np.array([[0],[0.01]])
        Kx = np.array([[150,15.0198]])
            
        ux = -Kx@(wx - ref_wx) #腳踝pitch控制x方向

        

        

        # Ay = np.array([[1,-0.00247],[-0.8832,1]])
        # By = np.array([[0],[0.01]])
        Ky = np.array([[-150,15]])

        #要補角動量切換！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

        # if self.stance_past == 0 and self.stance == 1:
        #     self.mea_y_L[1,0] = copy.deepcopy(self.mea_y_past_R[1,0])

        uy = -Ky@(wy - ref_wy) #腳踝row控制x方向

        if cf != cf_past:
            ux = uy = 0

        #--torque assign
        torque_ankle_cf = - np.vstack(( ux, uy ))


        # if stance == 1:
        #     alip_x_data = np.array([[ref_x_L[0,0]],[ref_x_L[1,0]],[self.ob_x_L[0,0]],[self.ob_x_L[1,0]]])
        #     alip_y_data = np.array([[ref_y_L[0,0]],[ref_y_L[1,0]],[self.ob_y_L[0,0]],[self.ob_y_L[1,0]]])
        #     self.alip_x_publisher.publish(Float64MultiArray(data=alip_x_data))
        #     self.alip_y_publisher.publish(Float64MultiArray(data=alip_y_data))

        return torque_ankle_cf

class Outterloop:
    
    @classmethod
    def get_jv_cmd(cls, frame: RobotFrame, ref_pa_in_wf, jv, stance, state):
        
        pa_in_pf = {
            'pel': frame.pa_pel_in_pf,
            'lf': frame.pa_lf_in_pf,
            'rf': frame.pa_rf_in_pf
        }
        
        endVel = cls.__endErr_to_endVel(ref_pa_in_wf, pa_in_pf, frame.eularToGeo)
        
        return cls.__endVel_to_jv(state, stance, jv, endVel, frame.getJacobian())

    @staticmethod           
    def __endErr_to_endVel(ref_pa_in_wf, pa_in_pf, eularToGeo):
        '''外環減法器與P gain, 並轉成幾何角速度'''
        #========求相對骨盆的向量========#
        ref_pa_pelTOft_in_wf = {
            'lf': ref_pa_in_wf['lf'] - ref_pa_in_wf['pel'],
            'rf': ref_pa_in_wf['rf'] - ref_pa_in_wf['pel']
        }

        pa_pelTO_ft_in_pf = {
            'lf': pa_in_pf['lf'] - pa_in_pf['pel'],
            'rf': pa_in_pf['rf'] - pa_in_pf['pel'],
        }
        
        #========經加法器算誤差========#
        err_pa_pelTOft_in_pf = {
            'lf': ref_pa_pelTOft_in_wf['lf'] - pa_pelTO_ft_in_pf['lf'],
            'rf': ref_pa_pelTOft_in_wf['rf'] - pa_pelTO_ft_in_pf['rf']
        }

        #========經P gain作為微分========#
        derr_pa_pelTOft_in_pf = {
            'lf': 20 * err_pa_pelTOft_in_pf['lf'],
            'rf': 20 * err_pa_pelTOft_in_pf['rf']
        }
        
        #========歐拉角速度轉幾何角速度========#
        w_pelTOft_in_pf = {
            'lf': eularToGeo['lf'] @ derr_pa_pelTOft_in_pf['lf'][3:],
            'rf': eularToGeo['rf'] @ derr_pa_pelTOft_in_pf['rf'][3:],
        }

        return {
            'lf': np.vstack(( derr_pa_pelTOft_in_pf['lf'][:3], w_pelTOft_in_pf['lf'] )),
            'rf': np.vstack(( derr_pa_pelTOft_in_pf['rf'][:3], w_pelTOft_in_pf['rf'] )),
        }

    @staticmethod
    def __endVel_to_jv(state, stance, joint_velocity, endVel: dict, J:dict):
        cf, sf = stance
        
        jv = {
            'lf': joint_velocity[:6],
            'rf': joint_velocity[6:]
        }
        jv_ankle = {
            'lf': jv['lf'][-2:],
            'rf': jv['rf'][-2:]
        }
        
        if state == 0 or state == 2 :
            cmd_jv = {
                'lf': np.linalg.pinv(J['lf']) @ endVel['lf'],
                'rf': np.linalg.pinv(J['rf']) @ endVel['rf']
            }
            
        elif state == 1: #or state == 2 :#真雙支撐
            #支撐腳膝上四關節: 控骨盆z, axyz
            #擺動腳膝上四關節: 控落點xyz, az
            tgt = {
                cf: slice(2, 6),
                sf: [0,1,2,5]
            }
            
            ctrlVel = {
                cf: endVel[cf][tgt[cf]],
                sf: endVel[sf][tgt[sf]]
            }
            J_ankle_to_ctrlVel = {
                cf: J[cf][tgt[cf], 4:],
                sf: J[sf][tgt[sf], 4:]
            }
            J_knee_to_ctrlVel = {
                cf: J[cf][tgt[cf], :4],
                sf: J[sf][tgt[sf], :4]
            }
            cmd_jv_knee = {
                cf: np.linalg.pinv(J_knee_to_ctrlVel[cf]) @ ( ctrlVel[cf] - J_ankle_to_ctrlVel[cf] @ jv_ankle[cf] ),
                sf: np.linalg.pinv(J_knee_to_ctrlVel[sf]) @ ( ctrlVel[sf] - J_ankle_to_ctrlVel[sf] @ jv_ankle[sf] )
            }
            cmd_jv = {
                cf: np.vstack(( cmd_jv_knee[cf], 0, 0 )),
                sf: np.vstack(( cmd_jv_knee[sf], 0, 0 ))
            }
        
        return cmd_jv
    
class Innerloop:
    
    @classmethod
    def balance(cls, jp, ros, stance, px_in_lf, px_in_rf):
        #balance the robot to initial state by p_control
        ref_jp = np.zeros((12,1))
        kp = np.vstack([ 2, 2, 4, 6, 6, 4 ]*2)
        
        l_leg_gravity, r_leg_gravity = cls.__calculate_gravity(ros, jp, stance, px_in_lf, px_in_rf)
        gravity = np.vstack(( l_leg_gravity, r_leg_gravity ))
        
        torque = kp * (ref_jp-jp) + gravity
        
        return torque
    
    @classmethod
    def innerloopDynamics(cls, joint_velocity, cmd_jv: dict, ros: ROSInterfaces, jp, stance, px_in_lf, px_in_rf):
        jv = {
            'lf': joint_velocity[:6],
            'rf': joint_velocity[6:]
        }
        
        l_leg_gravity, r_leg_gravity = cls.__calculate_gravity(ros, jp, stance, px_in_lf, px_in_rf)
        kl, kr = cls.__calculate_kp(stance)

        
        gravity = {
            'lf': l_leg_gravity,
            'rf': r_leg_gravity
        }
        kp = {
            'lf': kl,
            'rf': kr
        }


        return {
            'lf': kp['lf'] * (cmd_jv['lf'] - jv['lf']) + gravity['lf'],
            'rf': kp['rf'] * (cmd_jv['rf'] - jv['rf']) + gravity['rf']
        }
 
    @staticmethod
    def __calculate_gravity(ros: ROSInterfaces, joint_position, stance, px_in_lf, px_in_rf):
        jp = {
            'lf': joint_position[:6],
            'rf': joint_position[6:]
        }
        jp_from_ft = {
            'lf': np.vstack( -jp['lf'][::-1], jp['rf'] ),
            'rf': np.vstack( -jp['rf'][::-1], jp['lf'] )

        }
        
        jv_single_leg = ja_single_leg = np.zeros((6,1))
        jv_total = ja_total = np.zeros((12,1))
        
        #半邊單腳模型6*1
        gravity_single_ft = {
            'lf': np.vstack(
                -pin.rnea(ros.stance_l_model, ros.stance_l_data, -jp['lf'][::-1], jv_single_leg, ja_single_leg)[::-1],
                ),
            'rf': np.vstack(
                -pin.rnea(ros.stance_r_model, ros.stance_r_data, -jp['rf'][::-1], jv_single_leg, ja_single_leg)[::-1],
                )

        }
        gravity_singleft = np.vstack( gravity_single_ft['lf'], gravity_single_ft['rf'] )
        
        #腳底建起的整體模型12*1
        _gravity_from_ft = {
            'lf': pin.rnea(ros.bipedal_l_model, ros.bipedal_l_data, jp_from_ft['lf'], jv_total, ja_total),
            'rf': pin.rnea(ros.bipedal_r_model, ros.bipedal_r_data, jp_from_ft['rf'], jv_total, ja_total),
        }
        
        gravity_from_ft = {
            'lf': np.vstack( -_gravity_from_ft['lf'][:6][::-1],  _gravity_from_ft['lf'][6:]       ),
            'rf': np.vstack(  _gravity_from_ft['rf'][:6]      , -_gravity_from_ft['rf'][6:][::-1] )
        }

        Leg_gravity = (abs(px_in_lf[1,0])/0.1)*gravity_singleft + ((0.1-abs(px_in_lf[1,0]))/0.1)*gravity_from_ft['lf'] if abs(px_in_lf[1,0]) < abs(px_in_rf[1,0]) else\
                      (abs(px_in_rf[1,0])/0.1)*gravity_singleft + ((0.1-abs(px_in_rf[1,0]))/0.1)*gravity_from_ft['rf'] if abs(px_in_rf[1,0]) < abs(px_in_lf[1,0]) else\
                      gravity_single_ft

        l_leg_gravity = np.reshape(Leg_gravity[0:6,0], (6,1))
        r_leg_gravity = np.reshape(Leg_gravity[6:,0], (6,1))

        return l_leg_gravity, r_leg_gravity

    @staticmethod
    def __calculate_kp(stance):
        cf, sf = stance
        kp ={
            cf: np.vstack(( 1.2, 1.2, 1.2, 1.2, 1.2, 1.2 )),
            sf: np.vstack(( 1, 1, 1, 1, 1, 1 ))
        }

        return kp['lf'], kp['rf']
