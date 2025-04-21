import numpy as np
from scipy.signal import butter
#================ import library ========================#
from utils.config import Config
from numbers import Number
#========================================================#
def butter2(fn) -> tuple[np.ndarray]:
    '''回傳z^-1的轉移函數係數 num, den'''
    fs = 1 / Config.TIMER_PERIOD
    wn_nml = fn * 2 / fs
    return butter(2, wn_nml)
    
class Filter:
    """
    Filter 類別用於數位濾波器的實現。
        - 建構器:
            - 參數:
                num (list): 濾波器的分子係數。
                den (list): 濾波器的分母係數。
                
        - filt(u: Number | np.ndarray) -> float | np.ndarray:
            - 參數:
                u (Number | np.ndarray): 輸入信號，接受 column vector 或 Number。
            - 返回:
                float | np.ndarray: 濾波後的輸出信號。
    """
    
    def __init__(self, num: list, den: list):
        '''num, den 是以z^-1的係數'''
        self.__num : np.ndarray = np.vstack(( num )) / den[0]
        self.__den : np.ndarray = np.vstack(( den )) / den[0]

        self.__isStarted : bool = True
        
        # pasts從左到右是新到舊 k, k-1, k-2
        self.__u_pasts : np.ndarray = None
        self.__y_pasts : np.ndarray = None

        
    def filt(self, u: Number|np.ndarray) -> float|np.ndarray :
        ''' u只接受是column vector或num '''
        is_u_num = isinstance(u, Number)
        if  is_u_num:
            u = np.array( [[u]] )
        if self.__isStarted:
            self.__isStarted = False
            self.__u_pasts = np.zeros(( len(u), len(self.__num)-1 ))
            self.__y_pasts = np.zeros(( len(u), len(self.__den)-1 ))
        
        y = np.hstack(( u, self.__u_pasts )) @ self.__num - self.__y_pasts @ self.__den[1:]
        self.__y_pasts = np.hstack(( y, self.__y_pasts[:,:-1] ))
        self.__u_pasts = np.hstack(( u, self.__u_pasts[:,:-1] ))
        
        return y.item() if is_u_num else\
               y

class Diffter:
    """
    Diffter 類別利用差分計算輸入信號的離散時間微分。
    方法:
        diff(u):
            參數:
                u (NDArray | float): 當前的輸入信號值。
            返回:
                y (NDArray | float): 輸入信號的離散時間微分。
    """
    
    def __init__(self):
        self.__isStarted = True
        self.__u_p = None
        
    def diff(self, u : np.ndarray | float ) ->  np.ndarray | float:
        if self.__isStarted:
            self.__isStarted = False
            self.__u_p = 0*u
            
        y = ( u - self.__u_p ) / Config.TIMER_PERIOD
        self.__u_p = u
        return  y
    
class Dsp:
    #====================微分器====================#
    DIFFTER_JP = Diffter()
    DIFFTER_P_COM_IN_WF = Diffter()
    
    # DIFFTER_P_PEL_IN_LF = Diffter()
    # DIFFTER_P_PEL_IN_RF = Diffter()
    
    #====================濾波器====================#
    tf_end = butter2(15)
    
    FILTER_JP = Filter(*butter2(15))
    FILTER_JV = Filter(*butter2(30))
    
    FILTER_P_PEL_IN_WF = Filter(*tf_end)
    FILTER_P_LF_IN_WF = Filter(*tf_end)
    FILTER_P_RF_IN_WF = Filter(*tf_end)
    
    FILTER_V_COM_IN_WF = Filter(*tf_end)
    