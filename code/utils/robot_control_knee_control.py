#================ import library ========================#
from std_msgs.msg import Float64MultiArray 
import numpy as np; np.set_printoptions(precision=2)
import copy
from scipy.spatial.transform import Rotation as R

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

def alip_L(self, stance, torque, ref_pa_com_in_wf, ref_pa_lf_in_wf):
    p_com_in_wf = copy.deepcopy(self.P_COM_wf)
    p_lf_in_wf = copy.deepcopy(self.P_L_wf)

    #質心相對L frame的位置
    p_lfTOcom_in_wf = p_com_in_wf - p_lf_in_wf
    ref_p_lfTOcom_in_wf = ref_pa_com_in_wf[:3] - ref_pa_lf_in_wf[:3]

    #計算質心速度(v從世界座標下求出)
    self.CX_dot_L = (p_com_in_wf[0,0] - self.CX_past_L)/self.timer_period
    self.CX_past_L = p_com_in_wf[0,0]
    self.CY_dot_L = (p_com_in_wf[1,0] - self.CY_past_L)/self.timer_period
    self.CY_past_L = p_com_in_wf[1,0]

    #velocity filter
    self.Vx_L = 0.7408*self.Vx_past_L + 0.2592*self.CX_dot_past_L  #濾過後的速度(5Hz)
    self.Vx_past_L = self.Vx_L
    self.CX_dot_past_L =  self.CX_dot_L

    self.Vy_L = 0.7408*self.Vy_past_L + 0.2592*self.CY_dot_past_L  #濾過後的速度(5Hz)
    self.Vy_past_L = self.Vy_L
    self.CY_dot_past_L =  self.CY_dot_L

    #量測值
    Xc_mea, Yc_mea, _ = p_lfTOcom_in_wf.flatten()
    Ly_mea = 9*self.Vx_L*0.45
    Lx_mea = -9*self.Vy_L*0.45 #(記得加負號)
    self.mea_x_L = np.array([[Xc_mea],[Ly_mea]])
    self.mea_y_L = np.array([[Yc_mea],[Lx_mea]])
    
    #假設速度保持0
    ref_x_L = np.vstack(( ref_p_lfTOcom_in_wf[0,0], 0 ))
    ref_y_L = np.vstack(( ref_p_lfTOcom_in_wf[1,0], 0))
    
    #xc & ly model(m=9 H=0.45 Ts=0.01)
    Ax = np.array([[1,0.00247],[0.8832,1]])
    Bx = np.array([[0],[0.01]])
    Kx = np.array([[150,15.0198]])
    Lx = np.array([[0.1390,0.0025],[0.8832,0.2803]])
        
    self.ap_L = -Kx@(self.mea_x_L - ref_x_L)

    if self.stance_past == 0 and self.stance == 1:
        self.ap_L = 0

    torque[4,0] = -self.ap_L
    #----update
    self.mea_x_past_L = self.mea_x_L
    self.ap_past_L = self.ap_L

    Ay = np.array([[1,-0.00247],[-0.8832,1]])
    By = np.array([[0],[0.01]])
    Ky = np.array([[-150,15]])


    if self.stance_past == 0 and self.stance == 1:
        self.mea_y_L[1,0] = copy.deepcopy(self.mea_y_past_R[1,0])

    self.ar_L = -Ky@(self.mea_y_L - ref_y_L)

    if self.stance_past == 0 and self.stance == 1:
        self.ar_L = 0

    #--torque assign
    torque[5,0] = -self.ar_L
    self.mea_y_past_L = self.mea_y_L
    self.ar_past_L = self.ar_L

    # self.effort_publisher.publish(Float64MultiArray(data=torque))
    tl_data= np.array([[torque[4,0]],[torque[5,0]]])
    self.torque_L_publisher.publish(Float64MultiArray(data=tl_data))


    if stance == 1:
        alip_x_data = np.array([[ref_x_L[0,0]],[ref_x_L[1,0]],[self.ob_x_L[0,0]],[self.ob_x_L[1,0]]])
        alip_y_data = np.array([[ref_y_L[0,0]],[ref_y_L[1,0]],[self.ob_y_L[0,0]],[self.ob_y_L[1,0]]])
        # alip_x_data = np.array([[self.ref_x_L[0,0]],[self.ref_x_L[1,0]],[self.mea_x_L[0,0]],[self.mea_x_L[1,0]]])
        # alip_y_data = np.array([[self.ref_y_L[0,0]],[self.ref_y_L[1,0]],[self.mea_y_L[0,0]],[self.mea_y_L[1,0]]])
        self.alip_x_publisher.publish(Float64MultiArray(data=alip_x_data))
        self.alip_y_publisher.publish(Float64MultiArray(data=alip_y_data))
        
        # if state == 30:
        #     collect_data = [str(ref_x_L[0,0]),str(ref_x_L[1,0]),str(self.ob_x_L[0,0]),str(self.ob_x_L[1,0]),
        #                     str(ref_y_L[0,0]),str(ref_y_L[1,0]),str(self.ob_y_L[0,0]),str(self.ob_y_L[1,0])]
        #     csv_file_name = '/home/ldsc/collect/alip_data.csv'
        #     with open(csv_file_name, 'a', newline='') as csvfile:
        #         # Create a CSV writer object
        #         csv_writer = csv.writer(csvfile)
        #         # Write the data
        #         csv_writer.writerow(collect_data)

    return torque
    
    
    
    
    
    
    
    
    
    
    
    
    
    A_x = np.array([
        [1,0.00247],
        [0.8832,1]
    ])
    B_x = np.vstack(( 0,0.01 ))  
    K_x = np.array([[150,15.0198]])
    
    p_cfTOcom_in_wf = p_com_in_wf - p_ft_in_wf[cf]
    
    
    var_x = {
        'lf': np.vstack((  )), 
        'rf': np.vstack((  ))
    }
    u = - K_x @ (self.mea_x_L - ref_x_L)
    
    
    
    
    
    


   

    #量測值
    Xc_mea = PX_l[0,0]
    Ly_mea = 9*self.Vx_L*0.45
    Yc_mea = PX_l[1,0]
    Lx_mea = -9*self.Vy_L*0.45 #(記得加負號)
    self.mea_x_L = np.array([[Xc_mea],[Ly_mea]])
    self.mea_y_L = np.array([[Yc_mea],[Lx_mea]])
    
    #參考值(直接拿從online_planning來的)
    # ref_x_L = copy.deepcopy(self.ref_x_L)
    # ref_y_L = copy.deepcopy(self.ref_y_L)
    
    ref_x_L = np.vstack(( 0.0, 0.0 ))
    ref_y_L = np.vstack((-0.1, 0.0))
    
    #xc & ly model(m=9 H=0.45 Ts=0.01)
    
    # Kx = np.array([[184.7274,9.9032]])
    # Lx = np.array([[0.1427,-0.0131],[0.8989,0.1427]]) 
    #--compensator
    self.ob_x_L = Ax@self.ob_x_past_L + self.ap_past_L*Bx + Lx@(self.mea_x_past_L - Cx@self.ob_x_past_L)

    #由於程式邏輯 使得左腳在擺動過程也會估測 然而並不會拿來使用
    #為了確保支撐腳切換過程 角動量估測連續性
    if self.stance_past == 0 and self.stance == 1:
        self.mea_x_L[1,0] = copy.deepcopy(self.mea_x_past_R[1,0])
        self.ob_x_L[1,0] = copy.deepcopy(self.ob_x_past_R[1,0])

    #----calculate toruqe
    # self.ap_L = -Kx@(self.ob_x_L)  #(地面給機器人 所以使用時要加負號)
    # self.ap_L = -torque[4,0] #torque[4,0]為左腳pitch對地,所以要加負號才會變成地對機器人
    
    # self.ap_L = -Kx@(self.ob_x_L - ref_x_L)*0.5
    self.ap_L = -Kx@(self.mea_x_L - ref_x_L)

    # if self.ap_L >= 3:
    #     self.ap_L = 3
    # elif self.ap_L <= -3:
    #     self.ap_L =-3

    #切換瞬間 扭矩切成0 避免腳沒踩穩
    if self.stance_past == 0 and self.stance == 1:
        self.ap_L = 0

    #--torque assign
    torque[4,0] = -self.ap_L
    #----update
    self.mea_x_past_L = self.mea_x_L
    self.ob_x_past_L = self.ob_x_L
    self.ap_past_L = self.ap_L

    #yc & lx model
    Ay = np.array([[1,-0.00247],[-0.8832,1]])
    By = np.array([[0],[0.01]])
    Cy = np.array([[1,0],[0,1]])  
    #--LQR
    # Ky = np.array([[-177.0596,9.6014]])
    Ky = np.array([[-150,15]])
    
    Ly = np.array([[0.1288,-0.0026],[-0.8832,0.1480]])
    #--compensator
    self.ob_y_L = Ay@self.ob_y_past_L + self.ar_past_L*By + Ly@(self.mea_y_past_L - Cy@self.ob_y_past_L)

    #由於程式邏輯 使得左腳在擺動過程也會估測 然而並不會拿來使用，因此踩踏瞬間角動量來自上一時刻
    #為了確保支撐腳切換過程 角動量估測連續性
    if self.stance_past == 0 and self.stance == 1:
        self.mea_y_L[1,0] = copy.deepcopy(self.mea_y_past_R[1,0])
        self.ob_y_L[1,0] = copy.deepcopy(self.ob_y_past_R[1,0])

    #----calculate toruqe
    # self.ar_L = -Ky@(self.ob_y_L)
    # self.ar_L = -torque[5,0]#torque[5,0]為左腳roll對地,所以要加負號才會變成地對機器人
    
    # self.ar_L = -Ky@(self.ob_y_L - ref_y_L)*0.15
    self.ar_L = -Ky@(self.mea_y_L - ref_y_L)

    # if self.ar_L >= 3:
    #     self.ar_L =3
    # elif self.ar_L <= -3:
    #     self.ar_L =-3

    #切換瞬間 扭矩切成0 避免腳沒踩穩
    if self.stance_past == 0 and self.stance == 1:
        self.ar_L = 0

    #--torque assign
    torque[5,0] = -self.ar_L
    # torque[5,0] = 0
    #----update
    self.mea_y_past_L = self.mea_y_L
    self.ob_y_past_L = self.ob_y_L
    self.ar_past_L = self.ar_L

    # self.effort_publisher.publish(Float64MultiArray(data=torque))
    tl_data= np.array([[torque[4,0]],[torque[5,0]]])
    self.torque_L_publisher.publish(Float64MultiArray(data=tl_data))


    if stance == 1:
        alip_x_data = np.array([[ref_x_L[0,0]],[ref_x_L[1,0]],[self.ob_x_L[0,0]],[self.ob_x_L[1,0]]])
        alip_y_data = np.array([[ref_y_L[0,0]],[ref_y_L[1,0]],[self.ob_y_L[0,0]],[self.ob_y_L[1,0]]])
        # alip_x_data = np.array([[self.ref_x_L[0,0]],[self.ref_x_L[1,0]],[self.mea_x_L[0,0]],[self.mea_x_L[1,0]]])
        # alip_y_data = np.array([[self.ref_y_L[0,0]],[self.ref_y_L[1,0]],[self.mea_y_L[0,0]],[self.mea_y_L[1,0]]])
        self.alip_x_publisher.publish(Float64MultiArray(data=alip_x_data))
        self.alip_y_publisher.publish(Float64MultiArray(data=alip_y_data))
        
        # if state == 30:
        #     collect_data = [str(ref_x_L[0,0]),str(ref_x_L[1,0]),str(self.ob_x_L[0,0]),str(self.ob_x_L[1,0]),
        #                     str(ref_y_L[0,0]),str(ref_y_L[1,0]),str(self.ob_y_L[0,0]),str(self.ob_y_L[1,0])]
        #     csv_file_name = '/home/ldsc/collect/alip_data.csv'
        #     with open(csv_file_name, 'a', newline='') as csvfile:
        #         # Create a CSV writer object
        #         csv_writer = csv.writer(csvfile)
        #         # Write the data
        #         csv_writer.writerow(collect_data)

    return torque

def alip_R(self,stance_type,px_in_rf,torque_ALIP,com_in_rf,state):
    # print("ALIP_R")

    torque = copy.deepcopy(torque_ALIP) 
    stance = copy.deepcopy(stance_type) 
    #獲取量測值(相對於右腳腳底)
    # PX_r = copy.deepcopy(com_in_rf)
    com_in_wf = copy.deepcopy(self.P_COM_wf)
    rx_in_wf = copy.deepcopy(self.P_R_wf)
    PX_r = com_in_wf - rx_in_wf
    
    #計算質心速度
    self.CX_dot_R = (com_in_wf[0,0] - self.CX_past_R)/self.timer_period
    self.CX_past_R = com_in_wf[0,0]
    self.CY_dot_R = (com_in_wf[1,0] - self.CY_past_R)/self.timer_period
    self.CY_past_R = com_in_wf[1,0]

    #velocity filter
    self.Vx_R = 0.7408*self.Vx_past_R + 0.2592*self.CX_dot_past_R  #濾過後的速度(5Hz)
    self.Vx_past_R = self.Vx_R
    self.CX_dot_past_R =  self.CX_dot_R

    self.Vy_R = 0.7408*self.Vy_past_R + 0.2592*self.CY_dot_past_R  #濾過後的速度(5Hz)
    self.Vy_past_R = self.Vy_R
    self.CY_dot_past_R =  self.CY_dot_R

    #量測值
    Xc_mea = PX_r[0,0]
    Ly_mea = 9*self.Vx_R*0.45
    Yc_mea = PX_r[1,0]
    Lx_mea = -9*self.Vy_R*0.45 #(記得加負號)
    self.mea_x_R = np.array([[Xc_mea],[Ly_mea]])
    self.mea_y_R = np.array([[Yc_mea],[Lx_mea]])

    #參考值(直接拿從online_planning來的)
    ref_x_R = 0
    ref_y_R = 0.1
    # self.PX_ref = np.array([[0.0],[0.0],[0.57],[0.0],[0.0],[0.0]])
    # self.LX_ref = np.array([[0.0],[0.1],[0.0],[0.0],[0.0],[0.0]])
    # self.RX_ref = np.array([[0.0],[-0.1],[0.0],[0.0],[0.0],[0.0]])

    #xc & ly model(m=9 H=0.45 Ts=0.01)
    Ax = np.array([[1,0.00247],[0.8832,1]])
    Bx = np.array([[0],[0.01]])
    Cx = np.array([[1,0],[0,1]])  
    #--LQR
    Kx = np.array([[290.3274,15.0198]])
    Lx = np.array([[0.1390,0.0025],[0.8832,0.2803]]) 
    # Kx = np.array([[184.7274,9.9032]])
    # Lx = np.array([[0.1427,-0.0131],[0.8989,0.1427]]) 
    
    #--compensator
    self.ob_x_R = Ax@self.ob_x_past_R + self.ap_past_R*Bx + Lx@(self.mea_x_past_R - Cx@self.ob_x_past_R)

    #由於程式邏輯 使得右腳在擺動過程也會估測 然而並不會拿來使用
    #為了確保支撐腳切換過程 角動量估測連續性
    if self.stance_past == 1 and self.stance == 0:
        self.mea_x_R[1,0] = copy.deepcopy(self.mea_x_past_L[1,0])
        self.ob_x_R[1,0] = copy.deepcopy(self.ob_x_past_L[1,0])
    
    #----calculate toruqe
    # self.ap_R = -Kx@(self.ob_x_R)  #(地面給機器人 所以使用時要加負號)
    # self.ap_R = -torque[10,0] #torque[10,0]為右腳pitch對地,所以要加負號才會變成地對機器人
    self.ap_R = -Kx@(self.ob_x_R - ref_x_R)*0.5

    # if self.ap_R >= 3:
    #     self.ap_R =3
    # elif self.ap_R <= -3:
    #     self.ap_R =-3

    #切換瞬間 扭矩切成0 避免腳沒踩穩
    if self.stance_past == 1 and self.stance == 0:
        self.ap_R = 0
    
    #--torque assign
    torque[10,0] = -self.ap_R
    #----update
    self.mea_x_past_R = self.mea_x_R
    self.ob_x_past_R = self.ob_x_R
    self.ap_past_R = self.ap_R

    #yc & lx model
    Ay = np.array([[1,-0.00247],[-0.8832,1]])
    By = np.array([[0],[0.01]])
    Cy = np.array([[1,0],[0,1]])  
    #--LQR
    # Ky = np.array([[-290.3274,15.0198]])
    # Ly = np.array([[0.1390,-0.0025],[-0.8832,0.2803]])
    Ky = np.array([[-177.0596,9.6014]])
    Ly = np.array([[0.1288,-0.0026],[-0.8832,0.1480]])
    #--compensator
    self.ob_y_R = Ay@self.ob_y_past_R + self.ar_past_R*By + Ly@(self.mea_y_past_R - Cy@self.ob_y_past_R)

    #由於程式邏輯 使得右腳在擺動過程也會估測 然而並不會拿來使用
    #為了確保支撐腳切換過程 角動量估測連續性
    if self.stance_past == 1 and self.stance == 0:
        self.mea_y_R[1,0] = copy.deepcopy(self.mea_y_past_L[1,0])
        self.ob_y_R[1,0] = copy.deepcopy(self.ob_y_past_L[1,0])

    #----calculate toruqe
    # self.ar_R = -Ky@(self.ob_y_R)
    # self.ar_R = -torque[11,0]#torque[11,0]為右腳roll對地,所以要加負號才會變成地對機器人
    self.ar_R = -Ky@(self.ob_y_R - ref_y_R)*0.15

    #切換瞬間 扭矩切成0 避免腳沒踩穩
    if self.stance_past == 1 and self.stance == 0:
        self.ar_R = 0

    # if self.ar_R >= 3:
    #     self.ar_R =3
    # elif self.ar_R <= -3:
    #     self.ar_R =-3

    #--torque assign
    torque[11,0] = -self.ar_R
    #----update
    self.mea_y_past_R = self.mea_y_R
    self.ob_y_past_R = self.ob_y_R
    self.ar_past_R = self.ar_R

    return torque
