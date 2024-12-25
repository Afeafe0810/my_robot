from std_msgs.msg import Float64MultiArray 
import numpy as np; np.set_printoptions(precision=2)
import copy

from utils.config import Config

class RobotFrame:
    def __init__(self):
        #==========微分器==========#
        self.diffter_p_com_in_wf = Diffter()
        #==========濾波器==========#
        self.filter_v_com_in_wf = Filter()

class Filter:
    def __init__(self):
        self.__isStarted = True
        self.__u_p = None
        self.__y_p = None
        # self.__u_pp = None
        # self.__y_pp = None
        
    def filt(self, u):
        if self.__isStarted:
            self.__isStarted = False
            self.__u_p = 0*u
            self.__y_p = 0*u
        
        y = 0.7408 * self.__y_p + 0.2592 * self.__u_p
        #update
        self.__y_p = y
        self.__u_p = u
        
        return  y

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