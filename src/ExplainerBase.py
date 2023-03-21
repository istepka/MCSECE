import numpy as np
import numpy.typing as npt
import pandas as pd

class ExplainerBase:
    '''
    Base Explainer - abstract class
    '''
    def __init__(self) -> None:
        pass

    def fit() -> None:
        pass

    def get_counterfactuals(self) -> npt.NDArray | pd.DataFrame:
        pass