"""Base classes for all counterfactual generation explainers"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseExplainer(ABC):

    # TODO: add field with supported constraints per data?

    @abstractmethod
    def generate(self, x: pd.Series) -> pd.DataFrame:
        """Generate cfec for a single example x

        :param x: single example to generate cfec for
        :type x: pd.Series of shape (n_features,)

        :rtype: pd.DataFrame of shape (n_counterfactuals, n_features)
        """
        pass
