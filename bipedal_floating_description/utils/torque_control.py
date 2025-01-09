#================ import library ========================#
from std_msgs.msg import Float64MultiArray 
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
    def update_torque(cls, frame: RobotFrame, jp, ros, stance, stance_past, px_in_lf, px_in_rf, contact_lf, contact_rf, state, ref, jv):
        cf, sf = stance
        cf_past, sf_past = stance_past
        JLL, JRR = frame.left_leg_jacobian()
        ref_pa_pel_in_wf, ref_pa_lf_in_wf, ref_pa_rf_in_wf = ref['pel'], ref['lf'], ref['rf']
        
        
        if state == 0:
            torque = Innerloop.balance(jp, ros, cf, px_in_lf, px_in_rf, contact_lf, contact_rf, state)
        if state in [1, 2]:
            torque = cls.__kneecontrol(frame, ros, jp, cf, px_in_lf, px_in_rf, contact_lf, contact_rf, ref_pa_pel_in_wf, ref_pa_lf_in_wf, ref_pa_rf_in_wf, jv, stance, state, JLL, JRR)
            
            torque[sf][4:6] = cls.__swingAnkle_PDcontrol(sf, frame.r_lf_to_wf, frame.r_rf_to_wf)
            torque[cf][4:6] = cls.__alip_control(frame, cf, cf_past, frame.p_com_in_wf, frame.p_lf_in_wf, frame.p_rf_in_wf, ref_pa_pel_in_wf, ref_pa_lf_in_wf,ref_pa_rf_in_wf)
            torque = np.vstack(( torque['lf'], torque['rf'] ))
        return torque
    
    @staticmethod
    def __kneecontrol(frame, ros, jp, cf, px_in_lf, px_in_rf, contact_lf, contact_rf, ref_pa_pel_in_wf, ref_pa_lf_in_wf, ref_pa_rf_in_wf, jv, stance, state, JLL, JRR):
        VL, VR = Outterloop.get_jv_cmd(frame, ref_pa_pel_in_wf, ref_pa_lf_in_wf, ref_pa_rf_in_wf, jv, stance, state, JLL, JRR)
        return Innerloop.innerloopDynamics(jv, VL, VR, ros, jp, cf, px_in_lf, px_in_rf, contact_lf, contact_rf, state)
    
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
    def __alip_control(frame:RobotFrame, cf, cf_past, p_com_in_wf, p_lf_in_wf, p_rf_in_wf, ref_pa_com_in_wf, ref_pa_lf_in_wf, ref_pa_rf_in_wf):
        
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
    def get_jv_cmd(cls, frame: RobotFrame, ref_pa_pel_in_wf, ref_pa_lf_in_wf, ref_pa_rf_in_wf, jv_f, stance, state, JLL, JRR):
        
        Le_2, Re_2 = cls.__endErr_to_endVel(frame, ref_pa_pel_in_wf, ref_pa_lf_in_wf, ref_pa_rf_in_wf)
        
        return cls.__endVel_to_jv(Le_2, Re_2, jv_f, stance, state, JLL, JRR)

    @staticmethod           
    def __endErr_to_endVel(frame: RobotFrame, ref_pa_pel_in_wf, ref_pa_lf_in_wf, ref_pa_rf_in_wf):
        pa_pel_in_pf, pa_lf_in_pf , pa_rf_in_pf = frame.pa_pel_in_pf, frame.pa_lf_in_pf , frame.pa_rf_in_pf
        #========求相對骨盆的向量========#
        ref_pa_pelTOlf_in_pf = ref_pa_lf_in_wf -ref_pa_pel_in_wf
        ref_pa_pelTOrf_in_pf = ref_pa_rf_in_wf -ref_pa_pel_in_wf
        
        pa_pelTOlf_in_pf = pa_lf_in_pf -pa_pel_in_pf
        pa_pelTOrf_in_pf = pa_rf_in_pf -pa_pel_in_pf
        
        #========經加法器算誤差========#
        err_pa_pelTOlf_in_pf = ref_pa_pelTOlf_in_pf - pa_pelTOlf_in_pf
        err_pa_pelTOrf_in_pf = ref_pa_pelTOrf_in_pf - pa_pelTOrf_in_pf

        #========經P gain作為微分========#
        derr_pa_pelTOlf_in_pf = 20 * err_pa_pelTOlf_in_pf
        derr_pa_pelTOrf_in_pf = 20 * err_pa_pelTOrf_in_pf
        
        #========歐拉角速度轉幾何角速度========#
        w_pelTOlf_in_pf = frame.eularToGeo['lf'] @ derr_pa_pelTOlf_in_pf[3:]
        w_pelTOrf_in_pf = frame.eularToGeo['rf'] @ derr_pa_pelTOrf_in_pf[3:]
        
        vw_pelTOlf_in_pf = np.vstack(( derr_pa_pelTOlf_in_pf[:3], w_pelTOlf_in_pf ))
        vw_pelTOrf_in_pf = np.vstack(( derr_pa_pelTOrf_in_pf[:3], w_pelTOrf_in_pf ))

        return vw_pelTOlf_in_pf, vw_pelTOrf_in_pf

    @staticmethod
    def __endVel_to_jv(Le_2, Re_2, jv_f, stance, state, JLL, JRR):
        cf, sf = stance
        
        endVel = {'lf': Le_2, 'rf': Re_2}
        jv = {'lf': jv_f[:6], 'rf': jv_f[6:]}
        jv_ankle = {'lf': jv['lf'][-2:], 'rf': jv['rf'][-2:]}
        J = {
            'lf': JLL,
            'rf': JRR
        }
        
        if state == 0 or state == 2 :
            cmd_jv = {
                'lf': np.linalg.pinv(J['lf']) @ endVel['lf'],
                'rf': np.linalg.pinv(J['rf']) @ endVel['rf']
            }
            
        elif state == 1: #or state == 2 :#真雙支撐
            #===========================支撐腳膝上四關節: 控骨盆z, axyz==================================#
            #===========================擺動腳膝上四關節: 控落點xyz, az==================================#
            ctrlVel = {
                cf: endVel[cf][2:],
                sf: endVel[sf][[0,1,2,5]]
            }
            J_ankle_to_ctrlVel = {
                cf: J[cf][2:, 4:],
                sf: J[sf][[0,1,2,-1], 4:]
            }
            
            J_knee_to_ctrlVel = {
                cf: J[cf][2:, :4],
                sf: J[sf][[0,1,2,-1], :4]
            }
            
            cmd_jv_knee = {
                cf: np.linalg.pinv(J_knee_to_ctrlVel[cf]) @ ( ctrlVel[cf] - J_ankle_to_ctrlVel[cf] @ jv_ankle[cf] ),
                sf: np.linalg.pinv(J_knee_to_ctrlVel[sf]) @ ( ctrlVel[sf] - J_ankle_to_ctrlVel[sf] @ jv_ankle[sf] )
            }
            cmd_jv = {
                cf: np.vstack(( cmd_jv_knee[cf], 0, 0 )),
                sf: np.vstack(( cmd_jv_knee[sf], 0, 0 ))
            }
        
        return cmd_jv['lf'], cmd_jv['rf']
    
class Innerloop:
    
    @classmethod
    def balance(cls, jp, ros, cf, px_in_lf, px_in_rf, l_contact, r_contact, state):
        #balance the robot to initial state by p_control
        ref_jp = np.zeros((12,1))
        kp = np.vstack([ 2, 2, 4, 6, 6, 4 ]*2)
        
        l_leg_gravity, r_leg_gravity, *_ = cls.__gravity_compemsate(ros, jp, cf, px_in_lf, px_in_rf, l_contact, r_contact, state)
        gravity = np.vstack(( l_leg_gravity, r_leg_gravity ))
        
        torque = kp * (ref_jp-jp) + gravity
        
        return torque
    
    @classmethod
    def innerloopDynamics(cls, jv, vl_cmd, vr_cmd, ros: ROSInterfaces, joint_position, cf, px_in_lf, px_in_rf, l_contact, r_contact, state):
        cmd_v = np.vstack(( vl_cmd, vr_cmd ))
        l_leg_gravity, r_leg_gravity, kl,kr = cls.__gravity_compemsate(ros, joint_position, cf, px_in_lf, px_in_rf, l_contact, r_contact, state)
        
        gravity = np.vstack(( l_leg_gravity, r_leg_gravity ))
        kp = np.vstack(( kl,kr ))

        torque = kp * (cmd_v - jv) + gravity

        return {
            'lf': torque[:6],
            'rf': torque[6:]
        }
    
    @staticmethod
    def __gravity_compemsate(ros: ROSInterfaces, joint_position, cf, px_in_lf, px_in_rf, l_contact, r_contact, state):

        jp = {
            'lf': joint_position[:6],
            'rf': joint_position[6:]
        }
        jp_from_ft = {
            'lf': np.vstack(( -jp['lf'][::-1], jp['rf'] )),
            'rf': np.vstack(( -jp['rf'][::-1], jp['lf'] ))
        }
        jv_single = ja_single = np.zeros(( 6,1))
        jv_double = ja_double = np.zeros((12,1))
        
        #==========半邊單腳模型==========#
        _gravity_single_ft = {
            'lf': -pin.rnea(ros.stance_l_model, ros.stance_l_data, -jp['lf'][::-1], jv_single, (ja_single))[::-1],
            'rf': -pin.rnea(ros.stance_r_model, ros.stance_r_data, -jp['rf'][::-1], jv_single, (ja_single))[::-1]
        }
        gravity_single = np.vstack(( *_gravity_single_ft['lf'], *_gravity_single_ft['rf'] ))
        
        #==========腳底建起的模型==========#
        _gravity_from_ft = {
            'lf': pin.rnea(ros.bipedal_l_model, ros.bipedal_l_data, jp_from_ft['lf'], jv_double, (ja_double)),
            'rf': pin.rnea(ros.bipedal_r_model, ros.bipedal_r_data, jp_from_ft['rf'], jv_double, (ja_double)),
        }
        gravity_from_ft = {
            'lf': np.vstack(( *-_gravity_from_ft['lf'][5::-1], *_gravity_from_ft['lf'][6:]    )),
            'rf': np.vstack(( *_gravity_from_ft['rf'][6:]   , *-_gravity_from_ft['rf'][5::-1] )),
        }
        
        #==========加權==========#
        weighted = lambda x, x0, x1, g0, g1 :\
            g0 +(g1-g0)/(x1-x0)*(x-x0)
        
        Leg_gravity = weighted(px_in_lf[1,0], *[0, -0.1], *[gravity_from_ft['lf'], gravity_single]) if abs(px_in_lf[1,0]) < abs(px_in_rf[1,0]) else\
                      weighted(px_in_rf[1,0], *[0,  0.1], *[gravity_from_ft['rf'], gravity_single]) if abs(px_in_lf[1,0]) < abs(px_in_rf[1,0]) else\
                      gravity_single
        
        if cf == 'rf':
            if r_contact == 1:
                kr = np.array([[1.2],[1.2],[1.2],[1.2],[1.2],[1.2]])
            else:
                kr = np.array([[1],[1],[1],[1],[1],[1]])
            if l_contact == 1:
                kl = np.array([[1.2],[1.2],[1.2],[1.2],[1.2],[1.2]])
            else:
                kl = np.array([[1],[1],[1],[1],[1],[1]])

        
        elif cf == 'lf':
            if r_contact == 1:
                kr = np.array([[1.2],[1.2],[1.2],[1.2],[1.2],[1.2]])
            else:
                kr = np.array([[1],[1],[1],[1],[1],[1]])
            if l_contact == 1:
                kl = np.array([[1.2],[1.2],[1.2],[1.2],[1.2],[1.2]])
            else:
                kl = np.array([[1],[1],[1],[1],[1],[1]])



        if state == 1:
            kr = np.array([[0.5],[0.5],[0.5],[0.5],[0.5],[0.5]])
            kl = np.array([[0.5],[0.5],[0.5],[0.5],[0.5],[0.5]])
        
        if state == 2:
            kr = np.array([[0.5],[0.5],[0.5],[0.5],[0.5],[0.5]])
            kl = np.array([[0.5],[0.5],[0.5],[0.5],[0.5],[0.5]])

        l_leg_gravity = np.reshape(Leg_gravity[0:6,0],(6,1))
        r_leg_gravity = np.reshape(Leg_gravity[6:,0],(6,1))

        return l_leg_gravity,r_leg_gravity,kl,kr