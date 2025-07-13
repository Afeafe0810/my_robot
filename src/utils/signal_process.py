from typing import TypeVar, overload, Any
from numbers import Number

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.signal import butter

#================ import library ========================#
from src.utils.config import Config, Vec, Mat, Arr
#========================================================#

NumberOrArray = TypeVar('NumberOrArray', Number, NDArray[np.floating])

def butter2(fn: float) -> tuple[Vec, Vec]:
    '''回傳z^-1的轉移函數係數 num, den'''
    fs = 1 / Config.Ts
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
    
    def __init__(self, num: ArrayLike, den: ArrayLike):
        '''num, den 是以z^-1的係數'''
        _den = np.array(den)
        
        self._num : Vec = np.array(num) / _den[0]
        self._den : Vec = _den / _den[0]

        self._has_run: bool = False
        
        # pasts從左到右是新到舊 k, k-1, k-2
        self._u_pasts : Mat
        self._y_pasts : Mat
    
    def _vector_filt(self, input: Vec) -> Vec:
        if not self._has_run:
            self._has_run = True
            self._u_pasts: Vec = np.zeros((len(input), len(self._num)-1))
            self._y_pasts: Vec = np.zeros((len(input), len(self._den)-1))
            
        y: Vec = np.column_stack((input, self._u_pasts)) @ self._num - self._y_pasts @ self._den[1:]
        
        self._y_pasts = np.column_stack(( y, self._y_pasts[:,:-1] ))
        self._u_pasts = np.column_stack(( input, self._u_pasts[:,:-1] ))
        
        return y
    
    @overload
    def filt(self, input: float | np.floating) -> float: ...
    
    @overload
    def filt(self, input: Vec) -> Vec: ...
    
    def filt(self, input):
        if isinstance(input, Number):
            return self._vector_filt(np.array([input])).item()
        else:
            return self._vector_filt(input)


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
        self._has_run: bool = False
        self._u_p: Any
    
    @overload
    def diff(self, u: float) -> float: ...
    
    @overload
    def diff(self, u: Vec) -> Vec: ...
    
    def diff(self, u):
        if not self._has_run:
            self._has_run = True
            self._u_p = 0 * u
            
        y = ( u - self._u_p ) / Config.Ts
        self._u_p = u
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
    