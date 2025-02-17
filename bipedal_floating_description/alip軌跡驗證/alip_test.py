import numpy as np
import pandas as pd
from math import cosh, sinh, cos, sin, pi
import csv
#================ import library ========================#

# from utils.frame_kinermatic import RobotFrame
#========================================================#
import numpy as np
#================ import library ========================#

#========================================================#

class Config:
    '''機器人的參數、常數'''
    
    #取樣頻率
    TIMER_PERIOD = 0.01
    
    #訂閱到的關節順序，是錯的
    JNT_ORDER_SUB = (
        'L_Hip_Z', 'L_Hip_Y', 'L_Knee_Y', 'L_Ankle_Y', 
        'L_Ankle_X', 'R_Hip_X', 'R_Hip_Z', 'R_Knee_Y', 
        'R_Hip_Y', 'R_Ankle_Y', 'L_Hip_X', 'R_Ankle_X'
    )

    #實際上的關節順序
    JNT_ORDER_LITERAL = [
        'L_Hip_X', 'L_Hip_Z', 'L_Hip_Y', 'L_Knee_Y', 
        'L_Ankle_Y', 'L_Ankle_X', 'R_Hip_X', 'R_Hip_Z', 
        'R_Hip_Y', 'R_Knee_Y', 'R_Ankle_Y', 'R_Ankle_X'
    ]
    
    #URDF檔的路徑
    ROBOT_MODEL_DIR = "/home/ldsc/ros2_ws/src/bipedal_floating_description/urdf"
    
    #骨盆相對base的座標
    P_PEL_IN_BASE = np.vstack(( 0, 0, 0.598 ))
    
    #單腳支撐時間?
    DDT = 2
    
    #行走每步時間?
    STEP_TIMELENGTH = 0.5
    
    #機器人的物理模型
    MASS = 9
    IDEAL_Z_COM_IN_WF = 0.45
    IDEAL_Y_RFTOLF_IN_WF = 0.2
    GC = 9.81
    OMEGA = ( GC / IDEAL_Z_COM_IN_WF )**0.5
    STEP_HEIGHT = 0.02
    
    #腳踝關節限制
    FOOT_WIDTH = 0.04
    ANKLE_LIMIT = MASS * GC * FOOT_WIDTH #3.53 Nm
    
class AlipTraj:
    def __init__(self):
        self.t = 0.0
        H = Config.IDEAL_Z_COM_IN_WF
        #初值
        self.var0 = {
            'x': np.vstack((0, 0)),
            'y': np.vstack((-0.1, 0))
        }
        self.p0_ftTocom_in_wf = {
            'lf': np.vstack(( 0, -0.1, H )),
            'rf': np.vstack(( 0,  0.1, H ))
        }
        self.p0_ft_in_wf = {
            'lf': np.vstack(( 0, 0.1, 0 )),
            'rf': np.vstack(( 0, -0.1, 0 ))
        }
    
    def plan(self, frame, stance:list, des_vx_com_in_wf_2T ):
        # input('go')
        
        
        isJustStarted = ( self.t == 0 )
        isTimesUp = ( self.t  - Config.STEP_TIMELENGTH > 1e-8)
        
        #==========當一步踩完, 時間歸零+主被動腳交換==========#
        if isTimesUp:
            print('timesup', self.t)
            self.t =0
            stance.reverse()
            self.p1_ft_in_wf = {
                'lf': self.Ans['lf'],
                'rf': self.Ans['rf']
            }
            # print(self.p1_ft_in_wf)
            
        cf, sf = stance
        #==========踩第一步的時候, 拿取初值與預測==========#
        if isTimesUp:
            self.var0, self.p0_ftTocom_in_wf, self.p0_ft_in_wf = self.get_alipInitialDate()
        if isTimesUp or isJustStarted:
            # print("did i")
            #(側旋角動量Lx須用ref, 不然軌跡極怪)
            self.var0['y'][1,0] = self.__get_ref_timesUp_Lx(sf) #現在的參考的角動量是前一個的支撐腳的結尾
            self.ref_xy_swTOcom_in_wf_T = self.__sf_placement(stance, des_vx_com_in_wf_2T)
            # print(self.ref_xy_swTOcom_in_wf_T)
            
            

        #==========得到軌跡點==========#
        ref_p_cfTOcom_in_wf = self.__plan_com(stance)
        ref_p_sfTOcom_in_wf = self.__sf_trajFit(stance)
        
        #==========轉成wf==========#
        ref_p_com_in_wf = ref_p_cfTOcom_in_wf + self.p0_ft_in_wf[cf]
        ref_p_sf_in_wf = - ref_p_sfTOcom_in_wf + ref_p_com_in_wf
        
        #==========更新時間==========#
        self.t +=Config.TIMER_PERIOD
        
        self.Ans = {
            'com': ref_p_com_in_wf,
               cf: self.p0_ft_in_wf[cf],
               sf: ref_p_sf_in_wf
        }
        
        return self.Ans
    
    #==========主要演算法==========# 
    def __plan_com(self, stance):
        cf, sf = stance
        
        #==========參數==========#
        H = Config.IDEAL_Z_COM_IN_WF
        
        #==========狀態轉移矩陣==========#
        A = self.__get_alipMatA(self.t)
        
        #==========狀態方程==========#
        var = {
            'x': A['x'] @ self.var0['x'],
            'y': A['y'] @ self.var0['y'],
        }
        # print('this', var)
        return np.vstack(( var['x'][0,0], var['y'][0,0], H ))
       
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
        L1 = ref_L_com_1T = {
            'y': ( A['x'] @ self.var0['x'] ) [1,0],
            'x': ( A['y'] @ self.var0['y'] ) [1,0]
        }
        
        cf1 = np.vstack([
            ( A['x'] @ self.var0['x'] ) [0,0],
            ( A['y'] @ self.var0['y'] ) [0,0]
        ])
        
        #==========下步落地向量==========#
        ref_xy_swTOcom_in_wf_T = np.vstack(( 
            ( des_L_com_2T['y'] - cosh(w*T)*ref_L_com_1T['y'] ) /  ( m*H*w*sinh(w*T) ),
            ( des_L_com_2T['x'] - cosh(w*T)*ref_L_com_1T['x'] ) / -( m*H*w*sinh(w*T) )
        ))
        # print(ref_xy_swTOcom_in_wf_T)
        self.var1 = {
            'x': np.vstack((ref_xy_swTOcom_in_wf_T[0,0], L1['y'])),
            'y': np.vstack((ref_xy_swTOcom_in_wf_T[1,0], L1['x'])),
        }
        self.p1_ftTocom_in_wf = {
             cf: cf1 ,
             sf: np.vstack(( ref_xy_swTOcom_in_wf_T, 0 ))
        }
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
        # print(xy1_sfTocom_in_wf.flatten())
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

    def get_alipInitialDate(self):
        return self.var1, self.p1_ftTocom_in_wf, self.p1_ft_in_wf

test = AlipTraj()
data_list = []
stance = ['lf', 'rf']
for i in np.arange(0, Config.STEP_TIMELENGTH*10+Config.TIMER_PERIOD, Config.TIMER_PERIOD):
    output = test.plan(None, stance, 0.15)
    com = output['com'].flatten()
    lf = output['lf'].flatten()
    rf = output['rf'].flatten()
    
    # 添加到數據列表
    data_list.append({
        'time_step': i,
        'com_x': com[0],
        'com_y': com[1],
        'com_z': com[2],
        'lf_x': lf[0],
        'lf_y': lf[1],
        'lf_z': lf[2],
        'rf_x': rf[0],
        'rf_y': rf[1],
        'rf_z': rf[2],
    })

# 轉為 DataFrame
df = pd.DataFrame(data_list)

# 保存為 CSV
df.to_csv('alip_traj_output.csv', index=False)

print("數據已成功存儲為 'alip_traj_output.csv'")