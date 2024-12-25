#================ import library ========================#
from std_msgs.msg import Float64MultiArray 
import numpy as np; np.set_printoptions(precision=2)
import copy
from scipy.spatial.transform import Rotation as R

#================ import library ========================#
from utils.rc_frame_kinermatic import RobotFrame

def outterloop():
    pass

def endErr_to_endVel(self):
    ref_pa_pel_in_wf = copy.deepcopy(self.PX_ref) #wf
    ref_pa_lf_in_wf  = copy.deepcopy(self.LX_ref) #wf
    ref_pa_rf_in_wf  = copy.deepcopy(self.RX_ref) #wf
    pa_pel_in_pf = copy.deepcopy(self.PX) #pf
    pa_lf_in_pf  = copy.deepcopy(self.LX) #pf
    pa_rf_in_pf  = copy.deepcopy(self.RX) #pf
    
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
    w_pelTOlf_in_pf = self.L_Body_transfer @ derr_pa_pelTOlf_in_pf[3:]
    w_pelTOrf_in_pf = self.R_Body_transfer @ derr_pa_pelTOrf_in_pf[3:]
    
    vw_pelTOlf_in_pf = np.vstack(( derr_pa_pelTOlf_in_pf[:3], w_pelTOlf_in_pf ))
    vw_pelTOrf_in_pf = np.vstack(( derr_pa_pelTOrf_in_pf[:3], w_pelTOrf_in_pf ))

    return vw_pelTOlf_in_pf, vw_pelTOrf_in_pf

def endVel_to_jv(Le_2,Re_2,jv_f,stance_type,state, JLL, JRR):
    state = copy.deepcopy(state)
    stance = copy.deepcopy(stance_type)
    
    cf, sf = ('lf','rf') if stance == 1 else \
             ('rf','lf') # if stance == 0, 2
    
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

def innerloop():
    pass

def balance(jp,l_leg_gravity,r_leg_gravity):
    #balance the robot to initial state by p_control
    ref_jp = np.zeros((12,1))
    
    kp = np.vstack([ 2, 2, 4, 6, 6, 4 ]*2)
    
    gravity = np.vstack(( l_leg_gravity, r_leg_gravity ))
    
    torque = kp * (ref_jp-jp) + gravity
    
    return torque

def swing_leg(self,jv,vl_cmd,vr_cmd,l_leg_gravity,r_leg_gravity,kl,kr):
    cmd_v = np.vstack(( vl_cmd, vr_cmd ))
    gravity = np.vstack(( l_leg_gravity, r_leg_gravity ))
    
    kp = np.vstack(( kl,kr ))


    torque = np.zeros((12,1))
    torque = kp * cmd_v + gravity
    
    vcmd_data = np.array([[vl_cmd[0,0]],[vl_cmd[1,0]],[vl_cmd[2,0]],[vl_cmd[3,0]],[vl_cmd[4,0]],[vl_cmd[5,0]]])
    self.vcmd_publisher.publish(Float64MultiArray(data=vcmd_data))
    jv_collect = np.array([[jv[0,0]],[jv[1,0]],[jv[2,0]],[jv[3,0]],[jv[4,0]],[jv[5,0]]])
    self.velocity_publisher.publish(Float64MultiArray(data=jv_collect))#檢查收到的速度(超髒)

    return torque
  
def walking_by_ALIP(self, jv, vl_cmd, vr_cmd, l_leg_gravity, r_leg_gravity, kl, kr, r_lf_to_wf, r_rf_to_wf):

    _, *ayx_lf_in_wf = np.vstack(( R.from_matrix(r_lf_to_wf).as_euler('zyx', degrees=False) ))
    _, *ayx_rf_in_wf = np.vstack(( R.from_matrix(r_rf_to_wf).as_euler('zyx', degrees=False) ))
    ref_jp = np.zeros((2,1))
    cmd_v = np.vstack(( vl_cmd, vr_cmd ))
    gravity = np.vstack(( l_leg_gravity, r_leg_gravity ))
    kp = np.vstack(( kl, kr ))
    
    torque = kp * (cmd_v - jv) + gravity

    # ankle PD
    torque[4:6]   = 0.1 * ( ref_jp - ayx_lf_in_wf )
    torque[10:12] = 0.1 * ( ref_jp - ayx_rf_in_wf )
    
    vcmd_data = np.array([[vl_cmd[0,0]],[vl_cmd[1,0]],[vl_cmd[2,0]],[vl_cmd[3,0]],[vl_cmd[4,0]],[vl_cmd[5,0]]])
    self.vcmd_publisher.publish(Float64MultiArray(data=vcmd_data))
    jv_collect = np.array([[jv[0,0]],[jv[1,0]],[jv[2,0]],[jv[3,0]],[jv[4,0]],[jv[5,0]]])
    self.velocity_publisher.publish(Float64MultiArray(data=jv_collect))#檢查收到的速度(超髒)

    return torque


def alip_L(self, stance, torque, ref_pa_com_in_wf, ref_pa_lf_in_wf, frame:RobotFrame):
    p_com_in_wf = copy.deepcopy(self.P_COM_wf)
    p_lf_in_wf = copy.deepcopy(self.P_L_wf)

    #質心相對L frame的位置
    x_lfTOcom_in_wf, y_lfTOcom_in_wf = ( p_com_in_wf - p_lf_in_wf ) [0:2,0]
    ref_x_lfTOcom_in_wf, ref_y_lfTOcom_in_wf = ( ref_pa_com_in_wf - ref_pa_lf_in_wf ) [0:2,0]

    #計算質心速度(v從世界座標下求出)
    vx_com_in_wf, vy_com_in_wf = frame.filter_v_com_in_wf.filt(
        frame.diffter_p_com_in_wf.diff(p_com_in_wf) 
    ) [0:2,0]
    
    
    Ly_com_in_wf =  9 * vx_com_in_wf * 0.45
    Lx_com_in_wf = -9 * vy_com_in_wf * 0.45
    
    wx = np.vstack(( x_lfTOcom_in_wf, Ly_com_in_wf ))
    wy = np.vstack(( y_lfTOcom_in_wf, Lx_com_in_wf ))
    
    ref_wx = np.vstack(( ref_x_lfTOcom_in_wf, 0 ))
    ref_wy = np.vstack(( ref_y_lfTOcom_in_wf, 0 ))
    
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

    if self.stance_past == 0 and self.stance == 1:
        ux = 0
        uy = 0

    #--torque assign
    torque[4:6] = - np.vstack(( ux, uy ))

    # self.effort_publisher.publish(Float64MultiArray(data=torque))
    tl_data= np.array([[torque[4,0]],[torque[5,0]]])
    self.torque_L_publisher.publish(Float64MultiArray(data=tl_data))


    # if stance == 1:
    #     alip_x_data = np.array([[ref_x_L[0,0]],[ref_x_L[1,0]],[self.ob_x_L[0,0]],[self.ob_x_L[1,0]]])
    #     alip_y_data = np.array([[ref_y_L[0,0]],[ref_y_L[1,0]],[self.ob_y_L[0,0]],[self.ob_y_L[1,0]]])
    #     self.alip_x_publisher.publish(Float64MultiArray(data=alip_x_data))
    #     self.alip_y_publisher.publish(Float64MultiArray(data=alip_y_data))

    return torque
