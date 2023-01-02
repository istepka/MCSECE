from typing import Union
import numpy as np
from numpy.typing import ArrayLike


class Metric:

    def __init__(self) -> None:
        pass

    def eval(x: float, distance_fn: str) -> float:
        pass

    def evalMatrix(x: np.ndarray, distance_fn: str) -> np.ndarray:
        pass