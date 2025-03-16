import numpy as np; np.set_printoptions(precision=2)
from math import cosh, sinh, cos, sin, pi
from copy import deepcopy
#================ import library ========================#
from motion_planning.utils import Ref
from utils.frame_kinermatic import RobotFrame
from utils.config import Config
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
    
    def plan(self, frame: RobotFrame, des_vx_com_in_wf_2T: float, is_firmly: dict[str, bool]) -> Ref:
        print(f"{self.T_n = }")
        
        cf, sf = self.stance
        
        if self.T_n == Config.STEP_SAMPLELENGTH:
            if is_firmly[sf]: #時間到了, 若踩穩就繼續規劃下一點
                self.T_n = 0
                self.stance.reverse()
                return self.ordinary_plan(frame, des_vx_com_in_wf_2T)
            
            else: #時間到, 若沒踩穩就輸出同一點
                return self.ref
                
        else: #時間還沒到, 就繼續規劃下一點
            self.T_n += 1
            return self.ordinary_plan(frame, des_vx_com_in_wf_2T)
        
    def ordinary_plan(self, frame: RobotFrame, des_vx_com_in_wf_2T: float) -> Ref:
        cf, sf = self.stance
        
        #踩第一步的時候, 拿取初值與預測
        if self.isJustStarted:
            self.isJustStarted = False
            self.initialize_alipData()
            
        #換腳時重新量取三部位
        elif self.T_n == 0:
            self.update_alipData(frame)
            self.T_n += 1
            
        #踏步過程中, 只重新更新cf部位
        else:
            self.update_cfData_only(frame, self.stance)
            
        #==========規劃落地點==========#
        self.ref_xy_swTOcom_in_wf_T = self._sf_placement(self.stance, des_vx_com_in_wf_2T)

        #==========得到軌跡點==========#
        ref_p_cfTOcom_in_wf, ref_var = self._plan_com(self.stance)
        ref_xy_sfTOcom_in_wf, ref_z_sf_in_wf = self._sf_trajFit(self.stance, self.p0_in_wf[sf][2])
        
        #==========轉成wf==========#
        ref_p_com_in_wf = ref_p_cfTOcom_in_wf + self.p0_in_wf[cf]
        ref_xy_sf_in_wf = - ref_xy_sfTOcom_in_wf + ref_p_com_in_wf[:2]
        
        #==========強制將軌跡設成跟cf同x, 使得不前進==========#
        ref_p_com_in_wf[0,0] = self.p0_ftTOcom_in_wf[cf][0,0]

        ref_ft = {
            cf: self.p0_in_wf[cf],
            sf: np.vstack((self.p0_in_wf[cf][0,0], ref_xy_sf_in_wf[1,0], ref_z_sf_in_wf))
        }
            
        #==========輸出==========#
        _com = np.vstack(( ref_p_com_in_wf, np.zeros((3,1)) ))
        #其實骨盆的x,y不是這麼重要, 主要是控com的x,y
        _pel = np.vstack((
            _com[0,0],
            _com[1,0] - frame.p_com_in_wf[1,0] + frame.p_pel_in_wf[1,0],
            # _com[:2] - frame.p_com_in_wf[:2] + frame.p_pel_in_wf[:2],
            Config.IDEAL_Z_PEL_IN_WF,
            np.zeros((3,1))
        ))
        
        self.ref = Ref(
            pel = _pel,
            lf  = np.vstack((ref_ft['lf'] , np.zeros((3,1)))),
            rf  = np.vstack((ref_ft['rf'] , np.zeros((3,1)))),
            var = ref_var,
            com = _com #只是用來plot而已
        )
        
        return self.ref

    #==========會不斷重算的property==========# 
    @property
    def t(self) -> float:
        return self.T_n * Config.TIMER_PERIOD
    
    @property
    def p0_ftTOcom_in_wf(self):
        return {
            'lf' : self.p0_in_wf['com'] - self.p0_in_wf['lf'],
            'rf' : self.p0_in_wf['com'] - self.p0_in_wf['rf']
        }
    
    def set_var(self, Ly: float, Lx: float) -> dict[str, np.ndarray]:
        """var中的角動量多用理論值, 所以不要用frame_kinermatic回傳的"""
        cf, sf = self.stance
        
        return {
            'x': np.vstack((self.p0_ftTOcom_in_wf[cf][0,0], Ly)),
            'y': np.vstack((self.p0_ftTOcom_in_wf[cf][1,0], Lx))
        }
    
    def initialize_alipData(self):
        """
        更新
        self.p0_in_wf
        self.var0
        """
        H = Config.IDEAL_Z_COM_IN_WF
        y0_com = 0.08
        
        self.p0_in_wf = {
            'com': np.vstack([0, y0_com, H]),
            'lf' : np.vstack([0,  0.1 , 0]),
            'rf' : np.vstack([0, -0.1 , 0])
        }
        
        self.var0 = self.set_var(0, 0)

    def update_alipData(self, frame: RobotFrame):
        cf, sf = self.stance
        
        #拿取三部位真實的量測值
        self.p0_in_wf = {
            'com': deepcopy(frame.p_com_in_wf),
            'lf' : deepcopy(frame.p_lf_in_wf) ,
            'rf' : deepcopy(frame.p_rf_in_wf)
        }
        
        #強制將支撐腳的高度設成0
        self.p0_in_wf[cf][2, 0] = 0.0
        
        #將角動量連續化, 且不要往前走
        self.var0 = self.set_var(
            Ly = 0.0,
            Lx = self.ref.Lx
        )
    
    def update_cfData_only(self, frame: RobotFrame, stance: list[str]):
        cf, sf = stance
        
        self.p0_in_wf[cf] = deepcopy(frame.p_ft_in_wf[cf])
        self.p0_in_wf[cf][2, 0] = 0.0
        
        self.var0 = self.set_var(
                Ly = self.var0['x'][1,0],
                Lx = self.var0['y'][1,0]
            )
        
    #==========主要演算法==========# 
    def _plan_com(self, stance: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]] :
        """回傳ref_com與ref_var"""
        cf, sf = stance
        
        #==========參數==========#
        H = Config.IDEAL_Z_COM_IN_WF
        
        #==========狀態轉移矩陣==========#
        A = self._get_alipMatA(self.t)
        
        #==========狀態方程==========#
        ref_var = {
            'x': A['x'] @ self.var0['x'],
            'y': A['y'] @ self.var0['y'],
        }
        print(ref_var)
        return np.vstack(( ref_var['x'][0,0], ref_var['y'][0,0], H )), ref_var
       
    def _sf_placement(self, stance: list[str], des_vx_com_in_wf_2T: float) -> np.ndarray:
        """回傳擺動腳到質心的落點向量"""
        cf, sf = stance
        
        #==========機器人參數==========#
        m = Config.MASS
        H = Config.IDEAL_Z_COM_IN_WF
        W = Config.IDEAL_Y_RFTOLF_IN_WF
        h = Config.STEP_HEIGHT
        w = Config.OMEGA
        T = Config.STEP_TIMELENGTH
        
        #==========下兩步的理想角動量(那時的支撐腳是現在的擺動腳)==========#
        des_L_com_2T = {
            'y': m * des_vx_com_in_wf_2T * H,
            'x': self._get_ref_timesUp_Lx(sf)
        }
        
        #==========下步落地的角動量==========#
        A = self._get_alipMatA(T)
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
        xy0 = self.p0_ftTOcom_in_wf[sf][:2]
        xy1 = self.ref_xy_swTOcom_in_wf_T
        ref_xy_sfTOcom_in_wf = xy0 + (xy1 - xy0)/(-2) * ( cos(pi*ratioT)-1 )
        
        #z方向用拋物線, 最高點在h
        ref_z_sf_in_wf = self._parabolic_fit(self.t)
        
        return ref_xy_sfTOcom_in_wf, ref_z_sf_in_wf
    #==========工具function==========#
    @staticmethod
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
    
    @staticmethod
    def _get_ref_timesUp_Lx(cf:str) -> float:
        """給定支撐腳, 會回傳踏步(過T秒)瞬間的角動量"""
        
        m = Config.MASS
        H = Config.IDEAL_Z_COM_IN_WF
        W = Config.IDEAL_Y_RFTOLF_IN_WF
        w = Config.OMEGA
        T = Config.STEP_TIMELENGTH
        
        sign_Lx = 1 if cf =='lf' else\
                 -1 #if cf == 'rf'
        return sign_Lx * ( 0.5*m*H*W ) * ( w*sinh(w*T) ) / ( 1+cosh(w*T) )

    @staticmethod
    def _parabolic_fit(t: float) -> float:
        """改回原本的, 用0->max->0的拋物線就好"""
        h = Config.STEP_HEIGHT
        T = Config.STEP_SAMPLELENGTH
        
        return (h - 4*h*(t/T - 0.5)**2)
    
