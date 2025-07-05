#================ import library ========================#
import numpy as np; np.set_printoptions(precision=3, suppress=True)

#================ import library ========================#
from src.utils.frame_kinermatic import RobotFrame
from src.utils.robot_model import RobotModel
from src.motion_planning import Ref

#TODO 感覺這邊可以重構, 沒有self property, 且所有method幾乎都是static method
class KneeLoop:
    """KneeLoop 類別負責雙腳膝蓋的扭矩控制"""
    def __init__(self):
        pass
    
    def ctrl(self, ref: Ref, frame: RobotFrame, robot: RobotModel, jp: np.ndarray, jv: np.ndarray,state: float, stance: list[str], is_firmly: dict[str, bool]) -> dict[str, np.ndarray]:
        
        #==========外環==========#
        endVel = self._endErr_to_endVel(frame, ref)
        _cmd_jv= self._endVel_to_jv(frame, endVel, jv, state, stance)
        cmd_jv = np.vstack(( _cmd_jv['lf'], _cmd_jv['rf'] ))
        
        #==========內環==========#
        the_knee = [0, 1, 2, 3, 6, 7, 8, 9]
        
        matM = robot.pure_knee_inertia(jp, stance)
        kv = self._get_innerloop_K(state, stance, is_firmly)
        
        tauI = matM @ kv @ (cmd_jv - jv)[the_knee]
        
        tauG = robot.gravity(jp, state, stance, *frame.get_posture())[the_knee]
        
        torque = tauI + tauG

        if state == 1:
            torque = np.vstack([0.5, 0.5, 0.5, 0.5]*2) * (cmd_jv - jv)[the_knee] + tauG
        if state == 2:
            torque = np.vstack([1.5, 1.5, 1.5, 1.5] + [0.5, 0.5, 0.5, 0.5]) * (cmd_jv - jv)[the_knee] + tauG
        if state == 3:
            torque = np.vstack([1.5, 1.5, 1.5, 1.5] + [0.5, 0.5, 0.5, 0.5]) * (cmd_jv - jv)[the_knee] + tauG
        return {
            'lf': np.vstack((torque[:4])),
            'rf': np.vstack((torque[4:]))
        }

    @staticmethod           
    def _endErr_to_endVel(frame: RobotFrame, ref: Ref) -> dict[str,np.ndarray] :
        """端末位置經過減法器 + P control + 方向矩陣，轉成端末速度"""
        pa_pel_in_pf, pa_lf_in_pf , pa_rf_in_pf = frame.pa_pel_in_pf, frame.pa_lf_in_pf , frame.pa_rf_in_pf #HACK 學長用pf, 但照理來說應該要用wf
        #========求相對骨盆的向量========#
        ref_pa_pelTOlf_in_pf = ref.lf - ref.pel
        ref_pa_pelTOrf_in_pf = ref.rf - ref.pel
        
        pa_pelTOlf_in_pf = pa_lf_in_pf - pa_pel_in_pf
        pa_pelTOrf_in_pf = pa_rf_in_pf - pa_pel_in_pf
        
        #========經加法器算誤差========#
        err_pa_pelTOlf_in_pf = ref_pa_pelTOlf_in_pf - pa_pelTOlf_in_pf
        err_pa_pelTOrf_in_pf = ref_pa_pelTOrf_in_pf - pa_pelTOrf_in_pf

        #========經P gain作為微分========#
        derr_pa_pelTOlf_in_pf = 25 * err_pa_pelTOlf_in_pf
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
        
        JL, JR = frame.get_jacobian()
        J = {
            'lf': JL,
            'rf': JR
        }

        match state:
            # case 2: #HACK 之後改成case 1一樣的，現在還沒排除干擾
            #     cmd_jv = {
            #         'lf': np.linalg.pinv(J['lf']) @ endVel['lf'],
            #         'rf': np.linalg.pinv(J['rf']) @ endVel['rf']
            #     }

            case 1 | 2 | 3 | 30:
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
    def _get_innerloop_K(state: float, stance: list[str], is_firmly: dict[str, bool]) -> tuple[np.ndarray]:
        # HACK 之後gain要改
        match state:
            case 1:
                kl = np.array([1, 1, 1, 1])*60
                kr = np.array([1, 1, 1, 1])*50
                return np.diag(np.hstack((kl, kr)))
        
            case 2 | 3:
                kl = np.array([1.5, 1.5, 1.5, 1.5])
                kr = np.array([0.5, 0.5, 0.5, 0.5])
                return np.diag(np.hstack((kl, kr)))
                
            case 30:
                cf, sf = stance
                
                k_sf = np.array(( 1.3, 1.3, 1.3, 1.3))
                
                k_cf = np.array((1.5, 1.5, 1.5, 1.5)) if is_firmly[cf] else\
                       np.array((1.2, 1.2, 1.2, 1.2))
                
                k = {cf: k_cf, sf: k_sf}        
                return np.diag(np.hstack((k['lf'], k['rf'])))
                
