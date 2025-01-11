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
    def __init__(self):
        pass
    
    @classmethod
    def plan(cls, stance, des_vx_com_in_cf_2T, t):
        cf, sf = stance
        
        m = Config.MASS
        H = Config.IDEAL_Z_COM_IN_WF
        w = Config.OMEGA
        T = Config.STEP_TIMELENGTH
        W = Config.IDEAL_Y_RFTOLF_IN_WF
        h = Config.STEP_HEIGHT
        sign = -1 if sf == 'rf' else\
                1
        
        #下兩步的理想角動量
        des_Ly_com_in_cf_2T = m * des_vx_com_in_cf_2T * H
        des_Lx_com_in_cf_2T = sign * ( 0.5*m*H*W ) * ( w*sinh(w*T) ) / ( 1+cosh(w*T) ) 
        
        #現在的質心ref
        X0, Y0, p0_cf_in_wf, p0_sfTOcom_in_wf = cls.__getInitialData()
        
        var_x = cls.__getAlipMatA('x', t) @ X0
        var_y = cls.__getAlipMatA('y', t) @ Y0

        ref_p_com_in_cf = np.vstack(( var_x[0,0], var_y[0,0], H ))
        ref_p_com_in_wf = ref_p_com_in_cf + p0_cf_in_wf
        
        #預測的下一步的初始角動量
        pdc_Ly_com_in_cf_1T = var_x[1,0]
        pdc_Lx_com_in_cf_1T = var_y[1,0]
        
        #下一步的軌跡點
        pdc_xy_swTOcom_in_cf_T = np.vstack(( 
            ( des_Ly_com_in_cf_2T - cosh(w*T)*pdc_Ly_com_in_cf_1T ) / ( m*H*w*sinh(w*T) ),
            ( des_Lx_com_in_cf_2T - cosh(w*T)*pdc_Lx_com_in_cf_1T ) / -( m*H*w*sinh(w*T) )
        ))
        
        #一步內擬和出的軌跡
        ratio = t/T
        
        ref_p_sfTOcom_in_cf = np.vstack((
            0.5*((1+cos(pi*ratio))*p0_sfTOcom_in_wf[:2] + (1-cos(pi*ratio))*pdc_xy_swTOcom_in_cf_T),
            4*h*(ratio-0.5)**2 + (H-h)
        ))

        ref_p_sf_in_cf = ref_p_com_in_cf - ref_p_sfTOcom_in_cf
        ref_p_sf_in_wf = ref_p_sf_in_cf + p0_cf_in_wf
        
        return {
            'pel': ref_p_com_in_wf,
               cf: p0_cf_in_wf,
               sf: ref_p_sf_in_wf,
        }

    @classmethod
    def __getInitialData(stance,t):
        cf, sf = stance
        if t == 0:
            
        X0, Y0, p0_cf_in_wf, p0_sfTOcom_in_wf = cls.__getInitialData()
    
    @staticmethod
    def __getAlipMatA(axis:str, t:float) -> np.ndarray:
        """理想ALIP動態矩陣"""
        m = Config.MASS
        H = Config.IDEAL_Z_COM_IN_WF
        w = Config.OMEGA
        
        if axis == 'x':
            return np.array([
                [         cosh(w*t), sinh(w*t)/(m*H*w) ], 
                [ m*H*w * sinh(w*t), cosh(w*t)         ]
            ])
        elif axis == 'y':
            return np.array([
                [          cosh(w*t), -sinh(w*t)/(m*H*w) ],
                [ -m*H*w * sinh(w*t),  cosh(w*t)         ]
            ])
    
if __name__ == '__main__':
    t = 0
    stance = ['lf', 'rf']
    while  t<=20:
        if t>Config.STEP_TIMELENGTH:
            stance.reverse()
            t=0
        AlipTraj.plan(stance, 0.15, t)
        t += Config.TIMER_PERIOD
        