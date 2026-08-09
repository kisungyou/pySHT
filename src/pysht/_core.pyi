from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

class BuildInfo(TypedDict):
    version: str
    cpp_standard: int
    stable_abi: bool

def build_info() -> BuildInfo: ...
def pairwise_distances(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]: ...
def pairwise_squared_distances(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]: ...
