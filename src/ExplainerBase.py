import numpy as np
import numpy.typing as npt

class ExplainerBase:
    '''
    Base Explainer - abstract class
    '''
    def __init__(self) -> None:
        pass

    def fit() -> None:
        pass

    def get_counterfactuals(self, query_instance: npt.NDArray, predicted_class: int) -> npt.NDArray:
        pass