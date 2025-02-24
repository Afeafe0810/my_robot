import numpy as np
import pandas as pd
from math import cosh, sinh, cos, sin, pi
import csv
from dataclasses import dataclass
#================ import library ========================#
from config import Config
#========================================================#

H= Config.IDEAL_Z_COM_IN_WF


@dataclass
class Ref:
    pel : np.ndarray
    lf  : np.ndarray
    rf  : np.ndarray
    var : dict[str, np.ndarray]
    com : np.ndarray = None #沒有這麼重要
    need_push : bool = False #預設都是False
    
class AlipTraj:
    def __init__(self):
        #AlIP是否推完的追蹤
        self.alip_need_push : bool = True
        #時間
        self.t : float = 0.0
        
        #幾個sample -> 因為float加法會出現二進制誤差，所以t改用T_n*T來計算
        self.T_n :int = 0
        
        #剛啟動
        self.isJustStarted = True
        
        #初值
        self.var0 : dict[str, np.ndarray] = None
        self.p0_ftTocom_in_wf : dict[str, np.ndarray] = None
        self.p0_ft_in_wf : dict[str, np.ndarray] = None
        self.ref_xy_swTOcom_in_wf_T : np.ndarray = None
        
        #紀錄ref
        self.ref: Ref = None
    
    def plan(self, stance:list[str], des_vx_com_in_wf_2T: float ) -> Ref:
        
        self.t = self.T_n * Config.TIMER_PERIOD
        cf, sf = stance
        
        #==========踩第一步的時候, 拿取初值與預測==========#
        if self.isJustStarted: #如果剛從state 2啟動
            self.isJustStarted = False
            p_pel_in_wf = np.vstack([0,  0.09, H])
            self.p0_ft_in_wf = {
                'lf' : np.vstack([0,  0.1 , 0]),
                'rf' : np.vstack([0, -0.1 , Config.STEP_HEIGHT])
            }
            
            self.p0_ftTocom_in_wf = {
                'lf' : p_pel_in_wf - self.p0_ft_in_wf['lf'],
                'rf' : p_pel_in_wf - self.p0_ft_in_wf['rf']
            }
            
            self.var0 = {
                'x': np.vstack((self.p0_ftTocom_in_wf[cf][0,0], 0)),
                'y': np.vstack((self.p0_ftTocom_in_wf[cf][1,0], 0))
            }
            
            self.var0['y'][1,0] = Config.INITIAL_LX
            self.ref_xy_swTOcom_in_wf_T = self._sf_placement(stance, des_vx_com_in_wf_2T)
            print(f"{self.ref_xy_swTOcom_in_wf_T.T = }")
            
        elif self.T_n == 0: #如果剛從state 2啟動
            self.p0_ft_in_wf = {
                'lf' : self.ref.lf[:3],
                'rf' : self.ref.rf[:3]
            }
            self.p0_ftTocom_in_wf = {
                'lf' : self.ref.pel[:3] - self.p0_ft_in_wf['lf'],
                'rf' : self.ref.pel[:3] - self.p0_ft_in_wf['rf']
            }
            
            self.var0 = {
                'x': np.vstack((self.p0_ftTocom_in_wf[cf][0,0], 0)),
                'y': np.vstack((self.p0_ftTocom_in_wf[cf][1,0], 0))
            }
            
            self.var0['y'][1,0] = self.ref.var['y'][1, 0] #現在的參考的角動量是前一個的支撐腳的結尾
            self.ref_xy_swTOcom_in_wf_T = self._sf_placement(stance, des_vx_com_in_wf_2T)

        #==========得到軌跡點==========#
        ref_p_cfTOcom_in_wf, ref_var = self._plan_com(stance)
        ref_xy_sfTOcom_in_wf, ref_z_sf_in_wf = self._sf_trajFit(stance, self.p0_ft_in_wf[sf][2])
        
        #==========轉成wf==========#
        ref_p_com_in_wf = ref_p_cfTOcom_in_wf + self.p0_ft_in_wf[cf]
        ref_xy_sf_in_wf = - ref_xy_sfTOcom_in_wf + ref_p_com_in_wf[:2]
        
        ref_ft = {
            cf: self.p0_ft_in_wf[cf],
            sf: np.vstack((ref_xy_sf_in_wf, ref_z_sf_in_wf))
        }
        
        #==========更新時間與初值==========#
        print(f"Walk.t = {self.t:.2f}")
        if self.T_n == Config.STEP_SAMPLELENGTH:
            self.T_n = 0
            stance.reverse()
        else:
            self.T_n += 1
        
        self.ref = Ref(
            com = np.vstack((ref_p_com_in_wf, np.zeros((3,1)))),
            lf  = np.vstack((ref_ft['lf']   , np.zeros((3,1)))),
            rf  = np.vstack((ref_ft['rf']   , np.zeros((3,1)))),
            var = ref_var,
            pel = np.vstack((ref_p_com_in_wf, np.zeros((3,1))))
        )
        
        return self.ref
        
    # def doesNeedToPush(self, frame: RobotFrame, stance: list[str]) -> bool:
    #     if self.alip_need_push:
    #         Lx = frame.get_alipdata(stance)[0]['y'][1,0]
            
    #         if Lx >= Config.INITIAL_LX:
    #             self.alip_need_push = False
                
    #     return self.alip_need_push
    
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
        xy0_sfTocom_in_wf = self.p0_ftTocom_in_wf[sf][:2]
        xy1_sfTocom_in_wf = self.ref_xy_swTOcom_in_wf_T
        ref_xy_sfTOcom_in_wf = xy0_sfTocom_in_wf + (xy1_sfTocom_in_wf - xy0_sfTocom_in_wf)/(-2) * ( cos(pi*ratioT)-1 )
        
        #z方向用拋物線, 最高點在h
        # ref_z_sfTOcom_in_wf = H - (h - 4*h*(ratioT-0.5)**2)
        ref_z_sf_in_wf = self._parabolic_fit(self.t, h0, h)
        # return np.vstack(( ref_xy_sfTOcom_in_wf, ref_z_sfTOcom_in_wf ))
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
    def _parabolic_fit(t: float, h0: float, hmax: float) -> float:
        h0, hmax = sorted([h0, hmax])
        T = Config.STEP_TIMELENGTH
        
        a = - ( 2*T*hmax - T*h0 + 2*T*np.sqrt(hmax*(hmax - h0))) / (T**3)
        b = ( 2*T*hmax - 2*T*h0 + 2*T*np.sqrt(hmax*(hmax - h0))) / (T**2)
        c = h0
        
        return a * t**2 + b*t + c


def store_ref(ref_store: pd.DataFrame, t:float, ref_now : Ref):
    new_data = {
        "t": t,
        'pel_x': ref_now.pel[0, 0],
        'pel_y': ref_now.pel[1, 0],
        'pel_z': ref_now.pel[2, 0],

        'lf_x': ref_now.lf[0, 0],
        'lf_y': ref_now.lf[1, 0],
        'lf_z': ref_now.lf[2, 0],

        'rf_x': ref_now.rf[0, 0],
        'rf_y': ref_now.rf[1, 0],
        'rf_z': ref_now.rf[2, 0],

        'x': ref_now.var['x'][0, 0],
        'y': ref_now.var['y'][0, 0],
        
        'Ly': ref_now.var['x'][1, 0],
        'Lx': ref_now.var['y'][1, 0],
    }
    # 建立一筆資料的 DataFrame
    new_df = pd.DataFrame([new_data])
    # 使用 pd.concat 進行疊加
    ref_store = pd.concat([ref_store, new_df], ignore_index=True)
    return ref_store
    
    return ref_store
    
if __name__ == "__main__":
    alip = AlipTraj()
    stance = ['lf', 'rf']
    ref_store = pd.DataFrame(columns=[
        't',
        'pel_x', 'pel_y', 'pel_z',
        'lf_x', 'lf_y', 'lf_z',
        'rf_x', 'rf_y', 'rf_z',
        'x', 'y',
        'Ly', 'Lx'
    ])
    t = 0
    for _ in range(50*8):
        ref = alip.plan(stance, 0)
        ref_store = store_ref(ref_store, t, ref)
        
        t += Config.TIMER_PERIOD
    
    print(ref_store)
    ref_store.to_csv("alip_test_data.csv")