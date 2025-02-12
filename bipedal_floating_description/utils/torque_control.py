#================ import library ========================#
from std_msgs.msg import Float64MultiArray 
import numpy as np; np.set_printoptions(precision=5)
import copy
from scipy.spatial.transform import Rotation as R
import pinocchio as pin

#================ import library ========================#
from utils.ros_interfaces import ROSInterfaces
from utils.frame_kinermatic import RobotFrame
from utils.config import Config
from utils.signal_process import Dsp

class TorqueControl:
    def __init__(self):
        self.alip = AlipControl()
        
    def update_torque(self, frame: RobotFrame, jp, ros, stance, stance_past, px_in_lf, px_in_rf, contact_lf, contact_rf, state, ref, jv):
        cf, sf = stance
        cf_past, sf_past = stance_past
        JLL, JRR = frame.left_leg_jacobian()
        ref_pa_pel_in_wf, ref_pa_lf_in_wf, ref_pa_rf_in_wf = ref['pel'], ref['lf'], ref['rf']
        
        
        if state == 0:
            torque = Innerloop.balance(jp, ros, cf, px_in_lf, px_in_rf, contact_lf, contact_rf, state)
        if state in [1, 2, 30]:
            torque = self.__kneecontrol(frame, ros, jp, cf, px_in_lf, px_in_rf, contact_lf, contact_rf, ref_pa_pel_in_wf, ref_pa_lf_in_wf, ref_pa_rf_in_wf, jv, stance, state, JLL, JRR)
            
            torque[sf][4:6] = self.__swingAnkle_PDcontrol(sf, frame.r_lf_to_wf, frame.r_rf_to_wf)
            torque[cf][4:6] = self.alip.ctrl(frame, stance, stance_past, ref['var'])
            torque = np.vstack(( torque['lf'], torque['rf'] ))
        return torque
    
    @staticmethod
    def __kneecontrol(frame, ros, jp, cf, px_in_lf, px_in_rf, contact_lf, contact_rf, ref_pa_pel_in_wf, ref_pa_lf_in_wf, ref_pa_rf_in_wf, jv, stance, state, JLL, JRR):
        VL, VR = Outterloop.get_jv_cmd(frame, ref_pa_pel_in_wf, ref_pa_lf_in_wf, ref_pa_rf_in_wf, jv, stance, state, JLL, JRR)
        return Innerloop.innerloopDynamics(jv, VL, VR, ros, jp, stance, px_in_lf, px_in_rf, contact_lf, contact_rf, state)
    
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

class AlipControl:
    def __init__(self):
        #(k-1)的輸入
        self.u_p_lf : dict[str, float] = {
            'x' : 0.,
            'y' : 0.
        }
        self.u_p_rf : dict[str, float] = {
            'x' : 0.,
            'y' : 0.
        }
        
        #(k-1)的估測值
        self.var_e_p_lf : dict[str, np.ndarray] = {
            'x': np.zeros((2,1)),
            'y': np.zeros((2,1))
        }
        self.var_e_p_rf : dict[str, np.ndarray] = {
            'x': np.zeros((2,1)),
            'y': np.zeros((2,1))
        }
        
        #(k-1)的量測值
        self.var_p_lf : dict[str, np.ndarray] = {
            'x': np.zeros((2,1)),
            'y': np.zeros((2,1))
        }
        self.var_p_rf : dict[str, np.ndarray] = {
            'x': np.zeros((2,1)),
            'y': np.zeros((2,1))
        }
        
        
    def ctrl(self, frame:RobotFrame, stance, stance_past, ref_var):
        
        cf, sf = stance
        
        if stance != stance_past:
            self.update_initialValue(stance)
            
        #==========量測的狀態變數==========#
        var_cf, *_ = frame.get_alipdata(stance)
        
        #==========過去的變數==========#
        u_p = {'lf': self.u_p_lf, 'rf': self.u_p_rf}
        var_e_p = {'lf': self.var_e_p_lf, 'rf': self.var_e_p_rf}
        var_p = {'lf': self.var_p_lf, 'rf': self.var_p_rf}
        
        u_p_cf, var_e_p_cf, var_p_cf = u_p[cf], var_e_p[cf], var_p[cf]
        
        #==========離散的狀態矩陣==========#
        matA = {
            'x': np.array([
                [ 1,      0.00247],
                [ 0.8832, 1      ]
            ]),
            
            'y': np.array([
                [  1,     -0.00247],
                [ -0.8832, 1      ]
            ])
        }
        
        matB = {
            'x': np.vstack(( 0, 0.01 )),
            'y': np.vstack(( 0, 0.01 )),
        }
        
        # matK = {
        #     'x': np.array([ [ 150, 15.0198] ]),
        #     'y': np.array([ [-150, 15     ] ])
        # }

        # matL = {
        #     'x': np.array([
        #         [ 0.1390, 0.0025],
        #         [ 0.8832, 0.2803]
        #     ]),
            
        #     'y': np.array([
        #         [  0.1288, -0.0026 ],
        #         [ -0.8832,  0.1480 ]
        #     ])
        # }
        
        matK = {
            'x': np.array([[290.3274,15.0198]])*0.5,
            'y': np.array([[-177.0596,9.6014]])*0.15
        }
        
        matL = {
            'x': np.array([[0.1390,0.0025],[0.8832,0.2803]]),
            
            'y': np.array([[0.1288,-0.0026],[-0.8832,0.1480]])
        }
        
        
        #==========估測器補償==========#
        var_e_cf = {
            'x': matA['x'] @ var_e_p_cf['x'] + matB['x'] * u_p_cf['x'] + matL['x'] @ (var_p_cf['x'] - var_e_p_cf['x']),
            'y': matA['y'] @ var_e_p_cf['y'] + matB['y'] * u_p_cf['y'] + matL['y'] @ (var_p_cf['y'] - var_e_p_cf['y']),
        }
        
        #==========全狀態回授==========#
        u_cf = {
            'x': -matK['x'] @ ( var_e_cf['x']-ref_var['x'] ), #腳踝pitch控制x方向
            'y': -matK['y'] @ ( var_e_cf['y']-ref_var['y'] ), #腳踝row控制x方向
        }
        
        # u = {
        #     'x': -matK['x'] @ ( var['x']-ref_var['x'] ), #腳踝pitch控制x方向
        #     'y': -matK['y'] @ ( var['y']-ref_var['y'] ), #腳踝row控制x方向
        # }
        
        

        


        #要補角動量切換！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

        # if self.stance_past == 0 and self.stance == 1:
        #     self.mea_y_L[1,0] = copy.deepcopy(self.mea_y_past_R[1,0])

        

        if stance != stance_past:
            u_cf['x'] = u_cf['y'] = 0

        #--torque assign
        torque_ankle_cf = - np.vstack(( u_cf['x'], u_cf['y'] ))
        
        #==========更新值==========#
        var_e_p[cf].update(var_e_cf)
        var_p[cf].update(var_cf)

        return torque_ankle_cf

    def update_initialValue(self, stance):
        cf, sf = stance
        #==========過去的變數==========#
        u_p = {'lf': self.u_p_lf, 'rf': self.u_p_rf}
        var_e_p = {'lf': self.var_e_p_lf, 'rf': self.var_e_p_rf}
        var_p = {'lf': self.var_p_lf, 'rf': self.var_p_rf}
        
        #切換瞬間扭矩舊值為0
        u_p[cf].update({ key: 0.0 for key in u_p})
        
        #切換瞬間量測的角動量一樣
        var_e_p[cf]['x'][1] = var_e_p[sf]['x'][1]
        var_e_p[cf]['y'][1] = var_e_p[sf]['y'][1]
        
        #切換瞬間量測值和估測值相同
        var_e_p[cf].update(var_p[cf])
            
    
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
        
        if state == 0 or state == 2 or state == 30:
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
        
        l_leg_gravity, r_leg_gravity, *_ = cls.__calculateGravity(ros, jp, px_in_lf, px_in_rf)
        gravity = np.vstack(( l_leg_gravity, r_leg_gravity ))
        
        torque = kp * (ref_jp-jp) + gravity
        
        return torque
    
    @classmethod
    def innerloopDynamics(cls, jv, vl_cmd, vr_cmd, ros: ROSInterfaces, joint_position, stance, px_in_lf, px_in_rf, l_contact, r_contact, state):
        cmd_v = np.vstack(( vl_cmd, vr_cmd ))
        l_leg_gravity, r_leg_gravity = cls.__calculateGravity(ros, joint_position, px_in_lf, px_in_rf)
        kl, kr  = cls.__determineK(state, stance)
        
        gravity = np.vstack(( l_leg_gravity, r_leg_gravity ))
        kp = np.vstack(( kl,kr ))

        torque = kp * (cmd_v - jv) + gravity

        return {
            'lf': torque[:6],
            'rf': torque[6:]
        }
    
    @staticmethod
    def __calculateGravity(ros: ROSInterfaces, joint_position, px_in_lf, px_in_rf):

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

        l_leg_gravity = np.reshape(Leg_gravity[0:6,0],(6,1))
        r_leg_gravity = np.reshape(Leg_gravity[6:,0],(6,1))

        return l_leg_gravity,r_leg_gravity
    
    @staticmethod
    def __determineK(state, stance):
        cf, sf = stance
        # if cf == 'rf':
        #     if r_contact == 1:
        #         kr = np.array([[1.2],[1.2],[1.2],[1.2],[1.2],[1.2]])
        #     else:
        #         kr = np.array([[1],[1],[1],[1],[1],[1]])
        #     if l_contact == 1:
        #         kl = np.array([[1.2],[1.2],[1.2],[1.2],[1.2],[1.2]])
        #     else:
        #         kl = np.array([[1],[1],[1],[1],[1],[1]])

        
        # elif cf == 'lf':
        #     if r_contact == 1:
        #         kr = np.array([[1.2],[1.2],[1.2],[1.2],[1.2],[1.2]])
        #     else:
        #         kr = np.array([[1],[1],[1],[1],[1],[1]])
        #     if l_contact == 1:
        #         kl = np.array([[1.2],[1.2],[1.2],[1.2],[1.2],[1.2]])
        #     else:
        #         kl = np.array([[1],[1],[1],[1],[1],[1]])



        if state == 1:
            kr = np.array([[0.5],[0.5],[0.5],[0.5],[0.5],[0.5]])
            kl = np.array([[0.5],[0.5],[0.5],[0.5],[0.5],[0.5]])
        
        if state == 2 or state == 30:
            kr = np.array([[0.5],[0.5],[0.5],[0.5],[0.5],[0.5]])
            kl = np.array([[0.5],[0.5],[0.5],[0.5],[0.5],[0.5]])
            
        return kl, kr
