import numpy as np; np.set_printoptions(precision=2)
from math import cosh, sinh, cos, sin, pi
from copy import deepcopy
#================ import library ========================#
from bipedal_floating_description.motion_planning.utils import Ref
from bipedal_floating_description.utils.frame_kinermatic import RobotFrame
from bipedal_floating_description.utils.config import Config
#========================================================#

# state 30
class AlipTraj:
    def __init__(self):
        #主、被動腳
        self.stance = ['lf', 'rf']
        
        #幾個sample -> 因為float加法會出現二進制誤差，所以t改用T_n*T來計算
        self.T_n : int = 0
        
        #剛啟動
        self.isJustStarted = True
        
        #三部位的初值(質心, 左腳掌, 右腳掌)
        self.p0_in_wf : dict[str, np.ndarray] = None
        
        #初始的狀態變數
        self.var0 : dict[str, np.ndarray] = None
        
        #軌跡規劃
        self.ref_xy_swTOcom_in_wf_T : np.ndarray = None
        
        #紀錄ref
        self.ref: Ref = None
    
    @property
    def t(self) -> float:
        return self.T_n * Config.Ts
    
    def plan(self, frame: RobotFrame, des_vx_com_in_wf_2T: float, is_firmly: dict[str, bool], stance: list[str]) -> Ref:
        '''演算法對外的接口
        
        處理實際換腳的邏輯: 當踩穩才跟演算法同步換腳, 並且演算法才會繼續規劃
        
        '''
        cf, sf = stance
        
        should_switch = stance != self.stance
        
        #演算法已換腳且踩穩, 就將main邏輯的stance變得跟演算法一致
        if should_switch and is_firmly[sf]:
            stance.reverse()
        
        #演算法已換腳但實際還沒踩穩, 就輸出同個軌跡
        if should_switch and not is_firmly[sf]:
            return self.ref
        else:
            return self.algorithm_plan(frame, des_vx_com_in_wf_2T)
        
    def algorithm_plan(self, frame: RobotFrame, des_vx_com_in_wf_2T: float) -> Ref:
        cf, sf = self.stance
        
        self.update_alipData(frame, des_vx_com_in_wf_2T)
        print(f"{self.T_n = }")
        
        # #==========規劃落地點==========#
        # self.ref_xy_swTOcom_in_wf_T = self._sf_placement(self.stance, des_vx_com_in_wf_2T)

        #==========得到軌跡點==========#
        ref_p_cfTOcom_in_wf, ref_var = self._plan_com(self.stance)
        ref_xy_sfTOcom_in_wf, ref_z_sf_in_wf = self._sf_trajFit(self.stance, self.p0_ft_in_wf[sf][2])
        
        #==========轉成wf==========#
        ref_p_com_in_wf = ref_p_cfTOcom_in_wf + self.p0_ft_in_wf[cf]
        ref_xy_sf_in_wf = - ref_xy_sfTOcom_in_wf + ref_p_com_in_wf[:2]
        
        ref_ft = {
            cf: self.p0_ft_in_wf[cf],
            sf: np.vstack((self.p0_ft_in_wf[cf][0,0], ref_xy_sf_in_wf[1,0], ref_z_sf_in_wf))
        }
        
        #==========更新時間與初值==========#
        if self.T_n == Config.STEP_SAMPLELENGTH:
            self.T_n = 0
            self.stance.reverse()
        else:
            self.T_n += 1
        _pel = np.vstack((ref_ft[cf][0,0], ref_p_com_in_wf[1,0],Config.IDEAL_Z_PEL_IN_WF, np.zeros((3,1))))
        self.ref = Ref(
            com = ref_p_com_in_wf,
            lf  = np.vstack((ref_ft['lf']   , np.zeros((3,1)))),
            rf  = np.vstack((ref_ft['rf']   , np.zeros((3,1)))),
            var = ref_var,
            pel = _pel
        )
        
        return self.ref
    
    #==========主要演算法==========# 
    def update_alipData(self, frame: RobotFrame, des_vx_com_in_wf_2T: float):
        
        cf, sf = self.stance
        
        #踩第一步的時候, 拿取初值與預測
        if self.isJustStarted:
            self.isJustStarted = False
            
            H = Config.IDEAL_Z_COM_IN_WF
            
            p_pel_in_wf = np.vstack([0,  0.08, H])
            
            self.p0_ft_in_wf = {
                'lf' : np.vstack([0,  0.1 , 0]),
                'rf' : np.vstack([0, -0.1 , 0])
            }
            
            self.p0_ftTOcom_in_wf = {
                'lf' : p_pel_in_wf - self.p0_ft_in_wf['lf'],
                'rf' : p_pel_in_wf - self.p0_ft_in_wf['rf']
            }
            
            self.var0 = {
                'x': np.vstack((self.p0_ftTOcom_in_wf[cf][0,0], 0)),
                'y': np.vstack((self.p0_ftTOcom_in_wf[cf][1,0], 0))
            }
            
            self.var0['y'][1,0] = 0
            self.ref_xy_swTOcom_in_wf_T = self._sf_placement(self.stance, des_vx_com_in_wf_2T)
            
            
        #換腳時重新量取三部位
        elif self.T_n == 0:
            self.p0_ft_in_wf = deepcopy(frame.p_ft_in_wf)
            self.p0_ft_in_wf[cf][2, 0] = 0.0
            
            self.p0_ftTOcom_in_wf = {
                'lf' : frame.p_com_in_wf[:3] - self.p0_ft_in_wf['lf'],
                'rf' : frame.p_com_in_wf[:3] - self.p0_ft_in_wf['rf']
            }
            
            self.var0 = deepcopy({
                'x': np.vstack(( self.p0_ftTOcom_in_wf[cf][0,0], 0 )), #x方向角動量設成0
                'y': np.vstack(( self.p0_ftTOcom_in_wf[cf][1,0], self.ref.var['y'][1,0] )) #現在的參考的角動量是前一個的支撐腳的結尾
            })
            
            self.ref_xy_swTOcom_in_wf_T = self._sf_placement(self.stance, des_vx_com_in_wf_2T)
            
            self.T_n += 1
            
        #踏步過程中, 只重新更新cf部位
        else:
            self.p0_ft_in_wf[cf] = deepcopy(frame.p_ft_in_wf[cf])
            self.p0_ft_in_wf[cf][2, 0] = 0.0

    def _plan_com(self, stance: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]] :
        """回傳ref_com與ref_var"""
        cf, sf = stance
        
        #==========參數==========#
        H = Config.IDEAL_Z_COM_IN_WF
        
        #==========狀態轉移矩陣==========#
        A = _get_alipMatA(self.t)
        
        #==========狀態方程==========#
        ref_var = {
            'x': A['x'] @ self.var0['x'],
            'y': A['y'] @ self.var0['y'],
        }
        return np.vstack(( ref_var['x'][0,0], ref_var['y'][0,0], H )), ref_var
       
    def _sf_placement(self, stance: list[str], des_vx_com_in_wf_2T: float) -> np.ndarray:
        """回傳擺動腳到質心的落點向量"""
        cf, sf = stance
        
        #==========機器人參數==========#
        m = Config.MASS
        H = Config.IDEAL_Z_COM_IN_WF
        W = Config.IDEAL_Y_STEPLENGTH
        h = Config.STEP_HEIGHT
        w = Config.OMEGA
        T = Config.STEP_TIMELENGTH
        
        #==========下兩步的理想角動量(那時的支撐腳是現在的擺動腳)==========#
        des_L_com_2T = {
            'y': m * des_vx_com_in_wf_2T * H,
            'x': _get_ref_timesUp_Lx(sf)
        }
        
        #==========下步落地的角動量==========#
        A = _get_alipMatA(T)
        ref_L_com_1T = {
            'y': ( A['x'] @ self.var0['x'] ) [1,0],
            'x': ( A['y'] @ self.var0['y'] ) [1,0]
        }
        
        #==========下步落地向量==========#
        ref_xy_swTOcom_in_wf_T = np.vstack(( 
            ( des_L_com_2T['y'] - cosh(w*T)*ref_L_com_1T['y'] ) /  ( m*H*w*sinh(w*T) ),
            ( des_L_com_2T['x'] - cosh(w*T)*ref_L_com_1T['x'] ) / -( m*H*w*sinh(w*T) )
        ))
        
        return ref_xy_swTOcom_in_wf_T

    def _sf_trajFit(self, stance: list[str], h0: float) -> tuple[np.ndarray, float]:
        """
        擬合擺動腳的軌跡
            - xy方向用三角函數模擬簡諧
            - z方向用拋物線模擬重力
        """
        cf, sf = stance
        
        h = Config.STEP_HEIGHT
        H = Config.IDEAL_Z_COM_IN_WF
        T = Config.STEP_TIMELENGTH
        
        ratioT = self.t/T
        
        #x,y方向用三角函數進行線性擬合，有點像簡諧(初/末速=0)
        xy0_sfTocom_in_wf = self.p0_ftTOcom_in_wf[sf][:2]
        xy1_sfTocom_in_wf = self.ref_xy_swTOcom_in_wf_T
        ref_xy_sfTOcom_in_wf = xy0_sfTocom_in_wf + (xy1_sfTocom_in_wf - xy0_sfTocom_in_wf)/(-2) * ( cos(pi*ratioT)-1 )
        
        #z方向用拋物線, 最高點在h
        # ref_z_sfTOcom_in_wf = H - (h - 4*h*(ratioT-0.5)**2)
        ref_z_sf_in_wf = _parabolic_fit(self.t, h0, h)
        # return np.vstack(( ref_xy_sfTOcom_in_wf, ref_z_sfTOcom_in_wf ))
        return ref_xy_sfTOcom_in_wf, ref_z_sf_in_wf
    
#==========工具function==========#
def _get_alipMatA(t):
    m = Config.MASS
    H = Config.IDEAL_Z_COM_IN_WF
    w = Config.OMEGA
    return {
        'x': np.array([
                [         cosh(w*t), sinh(w*t)/(m*H*w) ], 
                [ m*H*w * sinh(w*t), cosh(w*t)         ]   
            ]),
        'y': np.array([
                [          cosh(w*t), -sinh(w*t)/(m*H*w) ],
                [ -m*H*w * sinh(w*t),  cosh(w*t)         ]
            ])
    }
    
def _get_ref_timesUp_Lx(cf:str) -> float:
    """給定支撐腳, 會回傳踏步(過T秒)瞬間的角動量"""
    
    m = Config.MASS
    H = Config.IDEAL_Z_COM_IN_WF
    W = Config.IDEAL_Y_STEPLENGTH
    w = Config.OMEGA
    T = Config.STEP_TIMELENGTH
    
    sign_Lx = 1 if cf =='lf' else\
                -1 #if cf == 'rf'
    return sign_Lx * ( 0.5*m*H*W ) * ( w*sinh(w*T) ) / ( 1+cosh(w*T) )

def _parabolic_fit(t: float, h0: float, hmax: float) -> float:
    h0, hmax = sorted([h0, hmax])
    T = Config.STEP_TIMELENGTH
    
    a = - ( 2*T*hmax - T*h0 + 2*T*np.sqrt(hmax*(hmax - h0))) / (T**3)
    b = ( 2*T*hmax - 2*T*h0 + 2*T*np.sqrt(hmax*(hmax - h0))) / (T**2)
    c = h0
    
    return a * t**2 + b*t + c
    
