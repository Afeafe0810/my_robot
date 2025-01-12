import numpy as np; np.set_printoptions(precision=2)
from math import cosh, sinh, cos, sin, pi
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
    # def __init__(self):
    #     t = 0
    #     stance = ['lf', 'rf']
    #     while  t<=20:
    #         if t>Config.STEP_TIMELENGTH:
    #             stance.reverse()
    #             t=0
    #         [
    #             self.ref_p_com_in_wf,
    #             self.p0_cf_in_wf,
    #             self.ref_p_sf_in_wf,
    #             self.ref_p_com_in_cf,
    #         ] = AlipTraj.plan(stance, 0.15, t)
    #         t += Config.TIMER_PERIOD
    
    def plan(self, stance, des_vx_com_in_cf_2T, t):
        cf, sf = stance
        
        m = Config.MASS
        H = Config.IDEAL_Z_COM_IN_WF
        w = Config.OMEGA
        T = Config.STEP_TIMELENGTH
        W = Config.IDEAL_Y_RFTOLF_IN_WF
        h = Config.STEP_HEIGHT
        
        #==========切換瞬間拿取初值==========#
        var0, p0_cf_in_wf, p0_sfTOcom_in_wf = self.__getInitialData()
        
        
        
        #==========規劃質心軌跡與骨盆軌跡==========#
        ref_p_com_in_cf, ref_p_com_in_wf = self.__plan_com(t, var0, p0_cf_in_wf)
        
        #==========規劃擺動腳軌跡==========#
        
        
        
        
        

        
        
        
        
        

        ref_p_sf_in_cf = ref_p_com_in_cf - ref_p_sfTOcom_in_cf
        ref_p_sf_in_wf = ref_p_sf_in_cf + p0_cf_in_wf
        
    # @classmethod
    # def __getInitialData(stance,t):
    #     cf, sf = stance
    #     if t == 0:
    #         return
    #     X0, Y0, p0_cf_in_wf, p0_sfTOcom_in_wf = cls.__getInitialData()
        
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
    
    @staticmethod
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
        
    def __plan_sf(self, t, stance, des_vx_com_in_cf_2T, var0: dict[str, np.ndarray], p0_sfTOcom_in_wf):
        cf, sf = stance
        
        m = Config.MASS
        H = Config.IDEAL_Z_COM_IN_WF
        W = Config.IDEAL_Y_RFTOLF_IN_WF
        h = Config.STEP_HEIGHT
        w = Config.OMEGA
        T = Config.STEP_TIMELENGTH
        
        
        sign_Lx = -1 if sf == 'rf' else\
                   1
                
        A = self.matA(t)
        
        #求出下兩步的理想角動量
        des_L_com_in_cf_2T = {
            'y': m * des_vx_com_in_cf_2T * H,
            'x': sign_Lx * ( 0.5*m*H*W ) * ( w*sinh(w*T) ) / ( 1+cosh(w*T) ) 
        }
        
        #下步落地的角動量
        ref_L_com_in_cf_1T = {
            'y': ( A['x'] @ var0['x'] ) [1, 0],
            'x': ( A['y'] @ var0['y'] ) [1, 0]
        }
        
        #下步落地向量
        ref_xy_swTOcom_in_cf_T = np.vstack(( 
            ( des_L_com_in_cf_2T['y'] - cosh(w*T)*ref_L_com_in_cf_1T['y'] ) /  ( m*H*w*sinh(w*T) ),
            ( des_L_com_in_cf_2T['x'] - cosh(w*T)*ref_L_com_in_cf_1T['x'] ) / -( m*H*w*sinh(w*T) )
        ))
        
        #一步內擬合出的軌跡
        ratio = t/T
        theta = cos(pi * ratio)
        angularMove = lambda a, x1, x0, a1, a0:\
            x0 + (x1-x0)/(a1-a0) * (a-a0)
        parabolicMove = lambda a, x1, x0, r1, r0:\
            
        
        ref_p_sfTOcom_in_cf = np.vstack((
            angularMove(theta, *[ ref_xy_swTOcom_in_cf_T, p0_sfTOcom_in_wf[:2] ], *[-1, 1] ),
            
            4*h*(ratio-0.5)**2 + (H-h)
        ))
        
        

    
        