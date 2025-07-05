import numpy as np; np.set_printoptions(precision=2)
from numpy.typing import NDArray
from typing import TypeVar

FA = TypeVar('FloatOrArray', float, NDArray)

def linear_move(Tn: int, T0: int, T1: int, p0: FA, p1: FA) -> FA:
    if Tn < T0:
        return p0
    elif Tn < T1:
        return ( p0*(T1-Tn) + p1*(Tn - T0) ) / (T1-T0)
    else:
        return p1
