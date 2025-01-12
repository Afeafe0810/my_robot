import numpy as np; np.set_printoptions(precision=2)
from math import cosh, sinh, cos, sin, pi
import csv
#================ import library ========================#
from utils.config import Config
from utils.frame_kinermatic import RobotFrame
#========================================================#

class Trajatory:
    def __init__(self):
        
        #state 2, 單腳舉起的時間
        self.leg_lift_time = 0
        
        #state 30, ALIP行走當下時間
        self.alip_time = 0
        
    def plan(self, state):
        if state in [0,1]: #假雙支撐, 真雙支撐
            return self.__bipedalBalanceTraj()
        
        elif state == 2: #骨盆移到支撐腳
            if self.leg_lift_time <= 10 * Config.DDT:
                self.leg_lift_time += Config.TIMER_PERIOD
                print("DS",self.leg_lift_time)
                
            return self.__comMoveTolf(self.leg_lift_time)
        
        elif state == 30: #ALIP規劃
            pass
    
    @staticmethod
    def __bipedalBalanceTraj():
        return {
            'pel': np.vstack(( 0,    0, 0.57, 0, 0, 0 )),
            'lf' : np.vstack(( 0,  0.1,    0, 0, 0, 0 )),
            'rf' : np.vstack(( 0, -0.1,    0, 0, 0, 0 )),
        }
    
    @staticmethod
    def __comMoveTolf(t):
        T = Config.DDT

        #==========線性移動==========#
        linearMove = lambda t, x0, x1, t0, t1:\
            np.clip(x0 + (x1-x0) * (t-t0)/(t1-t0), x0, x1 )
            
        y_pel = linearMove(t, *[0, 0.09], *[0*T, 0.5*T])
        z_sf  = linearMove(t, *[0, 0.05], *[1*T, 1.1*T])
        
        return {
            'pel': np.vstack(( 0, y_pel, 0.55, 0, 0, 0 )),
            'lf' : np.vstack(( 0,   0.1,    0, 0, 0, 0 )),
            'rf' : np.vstack(( 0,  -0.1, z_sf, 0, 0, 0 )),
        }

class AlipTraj:
    def __init__(self):
        # 初始化檔案名稱
        self.csv_filename = "output_data.csv"
        # 初始化 CSV 檔案並寫入標題行
        with open(self.csv_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["t", "com_x", "com_y", "com_z", "lf_x", "lf_y", "lf_z", "rf_x", "rf_y", "rf_z"])
            
    def play(self):
        t = 0
        stance = ['lf', 'rf']

        while t <= Config.STEP_TIMELENGTH:
            ref = self.plan(stance, 0.15, t)
            
            # 提取數據
            com = ref['com']
            lf = ref['lf']
            rf = ref['rf']

            # 將數據存入 CSV 檔案
            with open(self.csv_filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    t,
                    com[0][0], com[1][0], com[2][0],
                    lf[0][0], lf[1][0], lf[2][0],
                    rf[0][0], rf[1][0], rf[2][0]
                ])

            t += Config.TIMER_PERIOD
        
    def plan(self, stance, des_vx_com_in_cf_2T, t):
        cf, sf = stance
        
        #==========切換瞬間拿取初值==========#
        var0, p0_cf_in_wf, p0_sfTOcom_in_wf = self.__getInitialData(stance, t)

        #==========規劃質心軌跡與骨盆軌跡==========#
        ref_p_com_in_cf, ref_p_com_in_wf = self.__plan_com(t, var0, p0_cf_in_wf)
        
        #==========規劃擺動腳軌跡==========#
        ref_p_sf_in_wf = self.__plan_sf(t, stance, des_vx_com_in_cf_2T, var0, p0_sfTOcom_in_wf, p0_cf_in_wf, ref_p_com_in_cf)
        
        return {
            'com': ref_p_com_in_wf,
              cf : p0_cf_in_wf,
              sf : ref_p_sf_in_wf,
        }
        
    @staticmethod
    def __getInitialData(stance,t):
        #==初始狀態==
        H = Config.IDEAL_Z_COM_IN_WF
        var0 = {
            'x': np.vstack(( 0, 0 )),
            'y': np.vstack(( -0.1, 0))
        }
        p0_cf_in_wf = np.vstack(( 0, 0.1, 0 ))
        p0_sfTOcom_in_wf = np.vstack(( 0, 0.1, H))
        
        return var0, p0_cf_in_wf, p0_sfTOcom_in_wf
        
    @staticmethod
    def matA(t):
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
    
    def __plan_com(self, t, var0: dict[str, np.ndarray], p0_cf_in_wf: np.ndarray):
        H = Config.IDEAL_Z_COM_IN_WF
        A = self.matA(t)
        
        #state equation
        var = {
            'x': A['x'] @ var0['x'],
            'y': A['y'] @ var0['y'],
        }
        
        #質心在cf的軌跡
        ref_p_com_in_cf = np.vstack(( var['x'][0,0], var['y'][0,0], H ))
        
        #質心在wf的軌跡(理論直走的cf和wf姿態同向)
        ref_p_com_in_wf = ref_p_com_in_cf + p0_cf_in_wf
        
        #參考的角動量(cf腳踝要控)
        # ref_L_com_in_cf = np.vstack(( var['y'][1,0], var['x'][1,0] ))
        
        return ref_p_com_in_cf, ref_p_com_in_wf
        
    def __plan_sf(self, t, stance, des_vx_com_in_cf_2T, var0: dict[str, np.ndarray], p0_sfTOcom_in_wf, p0_cf_in_wf, ref_p_com_in_cf):
        cf, sf = stance
        
        #==========機器人參數==========#
        m = Config.MASS
        H = Config.IDEAL_Z_COM_IN_WF
        W = Config.IDEAL_Y_RFTOLF_IN_WF
        h = Config.STEP_HEIGHT
        w = Config.OMEGA
        T = Config.STEP_TIMELENGTH
        
        #==========狀態轉移矩陣==========#
        A = self.matA(t)
        
        #==========下兩步的理想角動量==========#
        sign_Lx = -1 if sf == 'rf' else\
                   1
                   
        des_L_com_in_cf_2T = {
            'y': m * des_vx_com_in_cf_2T * H,
            'x': sign_Lx * ( 0.5*m*H*W ) * ( w*sinh(w*T) ) / ( 1+cosh(w*T) ) 
        }
        
        #==========下步落地的角動量==========#
        ref_L_com_in_cf_1T = {
            'y': ( A['x'] @ var0['x'] ) [1,0],
            'x': ( A['y'] @ var0['y'] ) [1,0]
        }
        
        #==========下步落地向量==========#
        ref_xy_swTOcom_in_cf_T = np.vstack(( 
            ( des_L_com_in_cf_2T['y'] - cosh(w*T)*ref_L_com_in_cf_1T['y'] ) /  ( m*H*w*sinh(w*T) ),
            ( des_L_com_in_cf_2T['x'] - cosh(w*T)*ref_L_com_in_cf_1T['x'] ) / -( m*H*w*sinh(w*T) )
        ))
        
        #==========擬合一步內的軌跡==========#
        ratio = t/T
        
        #x,y方向用三角函數的線性擬合，有點像簡諧(初/末速=0)
        factor = cos(pi * ratio)
        
        angularMove = lambda factor, x0, x1:\
            x0 + (x1-x0)/(1-(-1)) * (factor-(-1))
        
        ref_xy_sfTOcom_in_cf = angularMove( factor, *[ p0_sfTOcom_in_wf[:2], ref_xy_swTOcom_in_cf_T] )
        
        #ratio從[0,1]的拋物線，最高點在h
        parabolicMove = lambda ratio:\
            h - 4*h*(ratio-0.5)**2
        
        ref_p_sf_in_cf = np.vstack((
            ref_p_com_in_cf[:2] - ref_xy_sfTOcom_in_cf, 
            parabolicMove(ratio)
        ))
        
        ref_p_sf_in_wf = ref_p_sf_in_cf + p0_cf_in_wf
        
        return ref_p_sf_in_wf
    