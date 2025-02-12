import numpy as np; np.set_printoptions(precision=2)
from math import cosh, sinh, cos, sin, pi
from copy import deepcopy
import csv
#================ import library ========================#
from utils.config import Config
from utils.frame_kinermatic import RobotFrame
#========================================================#

class Trajatory:
    def __init__(self):
        
        #state 2, 單腳舉起的時間
        self.leg_lift_time = 0
        
        #state 30
        self.aliptraj = AlipTraj()
        
    def plan(self, state, frame: RobotFrame, stance):
        
        if state in [0,1]: #假雙支撐, 真雙支撐
            ref = self.__bipedalBalanceTraj()
            ref['var'] = self.get_refvar_USING_pel(ref['pel'], ref['lf'])
            
        elif state == 2: #骨盆移到支撐腳
            if self.leg_lift_time <= 10 * Config.DDT:
                self.leg_lift_time += Config.TIMER_PERIOD
                print("DS",self.leg_lift_time)
                
            ref = self.__comMoveTolf(self.leg_lift_time)
            ref['var'] = self.get_refvar_USING_pel(ref['pel'], ref['lf'])
        
        elif state == 30: #ALIP規劃
            ref = self.aliptraj.plan(frame, stance, 0)
            #不確定是否可以真實的差距推導出ref
            ref['pel'] = np.vstack((
                ref['com'][:3] -frame.p_com_in_wf + frame.p_pel_in_wf,
                np.zeros((3,1))
            ))
        return ref
    
    @staticmethod
    def get_refvar_USING_pel(ref_pel, ref_cf):
        return {
            'x': np.vstack(( ref_pel[0]-ref_cf[0], 0 )),
            'y': np.vstack(( ref_pel[1]-ref_cf[1], 0 )),
        }
    
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
        self.t = 0.0
        
        #初值
        self.var0 = None
        self.p0_ftTocom_in_wf = None
        self.p0_ft_in_wf = None
    
    def plan(self, frame: RobotFrame, stance:list, des_vx_com_in_wf_2T ):
        isJustStarted = ( self.t == 0 )
        isTimesUp = ( self.t - Config.STEP_TIMELENGTH > 1e-8  )
        
        #==========當一步踩完, 時間歸零+主被動腳交換==========#
        if isTimesUp:
            self.t = 0
            stance.reverse()
            
        cf, sf = stance
        
        #==========踩第一步的時候, 拿取初值與預測==========#
        if isJustStarted or isTimesUp:
            self.var0, self.p0_ftTocom_in_wf, self.p0_ft_in_wf = frame.get_alipdata(stance)
            
            #(側旋角動量Lx須用ref, 不然軌跡極怪)
            self.var0['y'][1,0] = self.__get_ref_timesUp_Lx(sf) #現在的參考的角動量是前一個的支撐腳的結尾
            self.ref_xy_swTOcom_in_wf_T = self.__sf_placement(stance, des_vx_com_in_wf_2T)

        #==========得到軌跡點==========#
        ref_p_cfTOcom_in_wf, ref_var = self.__plan_com(stance)
        ref_p_sfTOcom_in_wf = self.__sf_trajFit(stance)
        
        #==========轉成wf==========#
        ref_p_com_in_wf = ref_p_cfTOcom_in_wf + self.p0_ft_in_wf[cf]
        ref_p_sf_in_wf = - ref_p_sfTOcom_in_wf + ref_p_com_in_wf
        
        #==========更新時間==========#
        self.t += Config.TIMER_PERIOD
        
        return deepcopy({
            'var': ref_var,
            'com': np.vstack((ref_p_com_in_wf, np.zeros((3,1)))),
               cf: np.vstack(( self.p0_ft_in_wf[cf], np.zeros((3,1)))),
               sf: np.vstack(( ref_p_sf_in_wf, np.zeros((3,1)))),
        })
        
    #==========主要演算法==========# 
    def __plan_com(self, stance):
        cf, sf = stance
        
        #==========參數==========#
        H = Config.IDEAL_Z_COM_IN_WF
        
        #==========狀態轉移矩陣==========#
        A = self.__get_alipMatA(self.t)
        
        #==========狀態方程==========#
        ref_var = {
            'x': A['x'] @ self.var0['x'],
            'y': A['y'] @ self.var0['y'],
        }
        
        return np.vstack(( ref_var['x'][0,0], ref_var['y'][0,0], H )), ref_var
       
    def __sf_placement(self, stance, des_vx_com_in_wf_2T):
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
            'x': self.__get_ref_timesUp_Lx(sf)
        }
        
        #==========下步落地的角動量==========#
        A = self.__get_alipMatA(T)
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

    def __sf_trajFit(self, stance):
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
        ref_z_sfTOcom_in_wf = H - (h - 4*h*(ratioT-0.5)**2)
        
        return np.vstack(( ref_xy_sfTOcom_in_wf, ref_z_sfTOcom_in_wf ))

    #==========工具function==========#
    @staticmethod
    def __get_alipMatA(t):
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
    def __get_ref_timesUp_Lx(cf):
        """給定支撐腳, 會回傳踏步(過T秒)瞬間的角動量"""
        
        m = Config.MASS
        H = Config.IDEAL_Z_COM_IN_WF
        W = Config.IDEAL_Y_RFTOLF_IN_WF
        w = Config.OMEGA
        T = Config.STEP_TIMELENGTH
        
        sign_Lx = 1 if cf =='lf' else\
                 -1 #if cf == 'rf'
        return sign_Lx * ( 0.5*m*H*W ) * ( w*sinh(w*T) ) / ( 1+cosh(w*T) )
