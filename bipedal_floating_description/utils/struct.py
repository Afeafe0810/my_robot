from numpy.typing import NDArray
from dataclasses import dataclass

@dataclass
class End:
    pel: NDArray
    lf: NDArray
    rf: NDArray