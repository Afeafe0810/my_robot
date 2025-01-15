import numpy as np
#================ import library ========================#
from utils.config import Config
#========================================================#
    
class Filter:
    def __init__(self, num: list, den: list):
        '''num, den 是以z^-1的係數'''
        self.__num = np.vstack(( num )) / den[0]
        self.__den = np.vstack(( den )) / den[0]

        self.__isStarted = True
        self.__u_past = None
        self.__y_past = None
        # self.__u_pp = None
        # self.__y_pp = None
        
    def filt(self, u):
        ''' u只接受是column vector或int '''
        u_type = type(u)
        if  u_type == int:
            u = np.array( [[u]] )
        if self.__isStarted:
            self.__isStarted = False
            self.__u_past = np.zeros(( len(u), len(self.__num)-1 ))
            self.__y_past = np.zeros(( len(u), len(self.__den)-1 ))
        
        y = np.hstack(( u, self.__u_past )) @ self.__num - self.__y_past @ self.__den[1:]
        self.__y_past = np.hstack(( y, self.__y_past[:,:-1] ))
        self.__u_past = np.hstack(( u, self.__u_past[:,:-1] ))
        
        return y.item() if u_type == int else\
               y

class Diffter:
    def __init__(self):
        self.__isStarted = True
        self.__u_p = None
        
    def diff(self, u):
        if self.__isStarted:
            self.__isStarted = False
            self.__u_p = 0*u
            
        y = ( u - self.__u_p ) / Config.TIMER_PERIOD
        self.__u_p = u
        return  y
    
class Dsp:
    #====================微分器====================#
    DIFFTER = {
        "p_com_in_wf" : Diffter(),
        "p_pel_in_lf" : Diffter(),
        "p_pel_in_rf" : Diffter(),
        "jp" : Diffter(),
    }
    DIFFTER_p_com_in_wf = Diffter()
    #====================濾波器====================#
    FILTER_v_com_in_wf = Filter([0, 0.2592], [1, -0.7408])
    FILTER = {
        "v_com_in_wf" : Filter([0, 0.2592], [1, -0.7408]),
        "v_pel_in_lf" : Filter([0, 0.2592], [1, -0.7408]),
        "v_pel_in_rf" : Filter([0, 0.2592], [1, -0.7408]),
        # "jp": Filter([0, 0.1453, 0.1078],[1, -1.1580, 0.4112]), #10Hz
        "jv": Filter([0, 1.014, -0.008067],[1, -0.0063, 0.0001383]), #100Hz
    }