#================ import library ========================#
import numpy as np; np.set_printoptions(precision=5)

#================ import library ========================#
from utils.ros_interfaces import RobotModel
from utils.frame_kinermatic import RobotFrame
from utils.motion_planning import Ref
from utils.config import Config

# TODO 碰撞偵測：正常的力 - 碰撞的力，再經過低通濾波器

class TorqueControl:
    """TorqueControl 類別負責處理機器人扭矩對state的pattern matching邏輯。"""
    
    def __init__(self):
        self.knee = KneeLoop()
        self.alip = AlipControl()
        
    def update_torque(self, frame: RobotFrame, robot: RobotModel, ref: Ref, state: float,
                      stance: list[str], stance_past: list[str],jp: np.ndarray, jv: np.ndarray) -> np.ndarray:
        
        cf, sf = stance
        
        match state:
            case 0:
                return balance_ctrl(frame, robot, jp)
            case 1 | 2 | 30:
                #雙腳膝蓋
                torque = self.knee.ctrl(ref, frame, robot, jp, jv, state, stance)
                
                #擺動腳腳踝
                torque[sf][4:6] = anklePD_ctrl(frame, sf)
                
                #支撐腳腳踝
                torque[cf][4:6] = self.alip.ctrl(frame, stance, stance_past, ref.var)
                
                if state == 30:
                    if ref.need_push:
                        torque[cf][5,0] = self.alip.init_push()
                    elif ref.need_close:
                        torque[cf][5,0] = 0
                        
                    print(f"u_cf3: {-torque[cf][5]}")
                return np.vstack(( torque['lf'], torque['rf'] ))
    
def anklePD_ctrl(frame: RobotFrame, sf: str):
    """支撐腳腳踝的PD控制"""
    ref_jp = np.zeros((2,1))

    r_ft_to_wf = {
        'lf': frame.r_lf_to_wf,
        'rf': frame.r_rf_to_wf
    }
    ayx_sf_in_wf = frame.rotMat_to_euler(r_ft_to_wf[sf]) [1:]
    
    torque_ankle_sf = 0.1 * ( ref_jp - ayx_sf_in_wf ) # HACK 現在只用P control
    
    return torque_ankle_sf

class AlipControl:
    """AlipControl 類別負責支撐腳腳踝用ALIP的狀態控制"""
    
    def __init__(self):
        # #(k-1)的輸入
        # self.u_p_lf : dict[str, float] = {
        #     'x' : 0.,
        #     'y' : 0.
        # }
        # self.u_p_rf : dict[str, float] = {
        #     'x' : 0.,
        #     'y' : 0.
        # }
        
        # #(k-1)的估測值
        # self.var_e_p_lf : dict[str, np.ndarray] = {
        #     'x': np.zeros((2,1)),
        #     'y': np.zeros((2,1))
        # }
        # self.var_e_p_rf : dict[str, np.ndarray] = {
        #     'x': np.zeros((2,1)),
        #     'y': np.zeros((2,1))
        # }
        
        # #(k-1)的量測值
        # self.var_p_lf : dict[str, np.ndarray] = {
        #     'x': np.zeros((2,1)),
        #     'y': np.zeros((2,1))
        # }
        # self.var_p_rf : dict[str, np.ndarray] = {
        #     'x': np.zeros((2,1)),
        #     'y': np.zeros((2,1))
        # }
        pass
    
    def init_push(self):
        """在 state2 切到 state30 的瞬間，把扭矩開到最大，給一個初始角動量"""
        return -Config.ANKLE_LIMIT #負號是關節順序相反
    
    def ctrl(self, frame:RobotFrame, stance: list[str], stance_past: list[str], ref_var: dict[str, np.ndarray]) -> np.ndarray:
        """回傳支撐腳腳踝扭矩"""
        cf, sf = stance
        
        # if stance != stance_past:
        #     self.update_initialValue(stance)
            
        #==========量測的狀態變數==========#
        var_cf, *_ = frame.get_alipdata(stance)
        
        #==========過去的變數==========#
        # u_p = {'lf': self.u_p_lf, 'rf': self.u_p_rf}
        # var_e_p = {'lf': self.var_e_p_lf, 'rf': self.var_e_p_rf}
        # var_p = {'lf': self.var_p_lf, 'rf': self.var_p_rf}
        
        # u_p_cf, var_e_p_cf, var_p_cf = u_p[cf], var_e_p[cf], var_p[cf]
        
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
        
        matK = {
            'x': np.array([ [ 150, 15.0198] ]),
            'y': np.array([ [-150, 15     ] ])
        }

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
        
        # matK = {
        #     'x': np.array([[290.3274,15.0198]])*0.5,
        #     'y': np.array([[-177.0596,9.6014]])*0.15
        # }
        
        matL = {
            'x': np.array([[0.1390,0.0025],[0.8832,0.2803]]),
            
            'y': np.array([[0.1288,-0.0026],[-0.8832,0.1480]])
        }
        
        
        #==========估測器補償==========#
        # var_e_cf = {
        #     'x': matA['x'] @ var_e_p_cf['x'] + matB['x'] * u_p_cf['x'] + matL['x'] @ (var_p_cf['x'] - var_e_p_cf['x']),
        #     'y': matA['y'] @ var_e_p_cf['y'] + matB['y'] * u_p_cf['y'] + matL['y'] @ (var_p_cf['y'] - var_e_p_cf['y']),
        # }
        
        #==========全狀態回授==========#
        u_cf : dict[str, np.ndarray] = {
            'y': -matK['x'] @ ( var_cf['x'] - ref_var['x'] ), #腳踝pitch控制x方向
            'x': -matK['y'] @ ( var_cf['y'] - ref_var['y'] ), #腳踝row控制y方向
        }
        
        print(f"u_cf1: {u_cf['x']}")
        u_cf['x'] = u_cf['x'].clip(-Config.ANKLE_LIMIT, Config.ANKLE_LIMIT) #飽和
        print(f"u_cf2: {u_cf['x']}")

        #要補角動量切換！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

        # if self.stance_past == 0 and self.stance == 1:
        #     self.mea_y_L[1,0] = copy.deepcopy(self.mea_y_past_R[1,0])

        

        # if stance != stance_past:
            # u_cf['x'] = u_cf['y'] = 0

        # ALIP計算的u指關節對軀體的扭矩, pub的torque是從骨盆往下建, 扭矩方向相反
        torque_ankle_cf = - np.vstack(( u_cf['y'], u_cf['x'] ))
        
        #==========更新值==========#
        # var_e_p[cf].update(var_e_cf)
        # var_p[cf].update(var_cf)

        return torque_ankle_cf

    # def update_initialValue(self, stance):
    #     cf, sf = stance
    #     #==========過去的變數==========#
    #     u_p = {'lf': self.u_p_lf, 'rf': self.u_p_rf}
    #     var_e_p = {'lf': self.var_e_p_lf, 'rf': self.var_e_p_rf}
    #     var_p = {'lf': self.var_p_lf, 'rf': self.var_p_rf}
        
    #     #切換瞬間扭矩舊值為0
    #     u_p[cf].update({ key: 0.0 for key in u_p})
        
    #     #切換瞬間量測的角動量一樣
    #     var_p[cf]['x'][1] = var_p[sf]['x'][1]
    #     var_p[cf]['y'][1] = var_p[sf]['y'][1]
        
    #     #切換瞬間估測值代入量測值
    #     var_e_p[cf].update(var_p[cf])
            
class KneeLoop:
    """KneeLoop 類別負責雙腳膝蓋的扭矩控制"""
    def __init__(self):
        pass
    
    def ctrl(self, ref: Ref, frame: RobotFrame, robot: RobotFrame, jp: np.ndarray, jv: np.ndarray,state: float, stance: list[str]) -> dict[str, np.ndarray]:
        
        #==========外環==========#
        endVel: dict[str, np.ndarray] = self._endErr_to_endVel(frame, ref)
        cmd_jv: dict[str, np.ndarray] = self._endVel_to_jv(frame, endVel, jv, state, stance)
        cmd_jv_ : np.ndarray = np.vstack(( cmd_jv['lf'], cmd_jv['rf'] ))
        
        #==========內環==========#
        tauG_lf, tauG_rf= frame.calculate_gravity(robot, jp)
        kl, kr  = self._get_innerloop_K(state, stance)
        
        tauG = np.vstack(( tauG_lf, tauG_rf ))
        kp = np.vstack(( kl,kr ))

        torque = kp * (cmd_jv_ - jv) + tauG

        return {
            'lf': torque[:6],
            'rf': torque[6:]
        }

    @staticmethod           
    def _endErr_to_endVel(frame: RobotFrame, ref: Ref) -> dict[str,np.ndarray] :
        """端末位置經過減法器 + P control + 方向矩陣，轉成端末速度"""
        pa_pel_in_pf, pa_lf_in_pf , pa_rf_in_pf = frame.pa_pel_in_pf, frame.pa_lf_in_pf , frame.pa_rf_in_pf
        #========求相對骨盆的向量========#
        ref_pa_pelTOlf_in_pf = ref.lf - ref.pel
        ref_pa_pelTOrf_in_pf = ref.rf - ref.pel
        
        pa_pelTOlf_in_pf = pa_lf_in_pf - pa_pel_in_pf
        pa_pelTOrf_in_pf = pa_rf_in_pf - pa_pel_in_pf
        
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

        return {
            'lf': vw_pelTOlf_in_pf,
            'rf': vw_pelTOrf_in_pf
        }

    @staticmethod
    def _endVel_to_jv(frame: RobotFrame, endVel: dict[str, np.ndarray], jv: np.ndarray, state: float, stance: list[str] ) -> dict[str, np.ndarray]:
        """端末速度映射到關節速度"""
        cf, sf = stance
        
        jv_ = {'lf': jv[:6], 'rf': jv[6:]}
        jv_ankle = {'lf': jv_['lf'][-2:], 'rf': jv_['rf'][-2:]}
        
        JL, JR = frame.left_leg_jacobian()
        J = {
            'lf': JL,
            'rf': JR
        }
        
        match state:
            case 2 | 30: #HACK 之後改成case 1一樣的，現在還沒排除干擾
                cmd_jv = {
                    'lf': np.linalg.pinv(J['lf']) @ endVel['lf'],
                    'rf': np.linalg.pinv(J['rf']) @ endVel['rf']
                }

            case 1:
                # 支撐腳膝上四關節: 控骨盆z, axyz  ；  擺動腳膝上四關節: 控落點xyz, az
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
        
        return cmd_jv
   
    @staticmethod
    def _get_innerloop_K(state: float, stance: list[str]) -> tuple[np.ndarray]:
        # HACK 之後gain要改
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

def balance_ctrl(frame: RobotFrame, robot:RobotModel, jp: np.ndarray) -> np.ndarray:
    """在剛開機狀態直接用關節角度的 單環 來平衡"""
    ref_jp = np.zeros((12,1))
    kp = np.vstack([ 2, 2, 4, 6, 6, 4 ]*2)
    
    tauG_lf, tauG_rf = frame.calculate_gravity(robot, jp)
    
    tauG = np.vstack(( tauG_lf, tauG_rf ))
    
    torque = kp * (ref_jp-jp) + tauG
    
    return torque