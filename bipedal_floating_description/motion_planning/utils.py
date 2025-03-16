from dataclasses import dataclass
import numpy as np

@dataclass
class Ref:
    pel : np.ndarray #[z, ax, ay, az]由cf膝上
    lf  : np.ndarray
    rf  : np.ndarray
    var : dict[str, np.ndarray]
    com : np.ndarray = None #[x,y]由cf腳踝ALIP #不重要, 只是用來畫圖, 功能由var取代
    
    @property
    def x(self):
        return self.var['x'][0,0]
    
    @property
    def Ly(self):
        return self.var['x'][1,0]
    
    @property
    def y(self):
        return self.var['y'][0,0]
    
    @property
    def Lx(self):
        return self.var['y'][1,0]
    
    @staticmethod
    def _generate_var_insteadOf_com(ref_pel: np.ndarray, ref_cf: np.ndarray) -> dict[str, np.ndarray] :
        """平衡(不管是單腳還是雙腳平衡)時, 用pel代替com, 角動量ref皆設0"""
        return {
            'x': np.vstack(( ref_pel[0]-ref_cf[0], 0 )),
            'y': np.vstack(( ref_pel[1]-ref_cf[1], 0 )),
        }
