from typing import List, Tuple, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

from cfec.constraints import ValueNominal, OneHot


def _stack_continuous_nominal(c: NDArray[np.float32], n: NDArray[np.float32]) -> NDArray[np.float32]:
    if len(c.shape) < 2:
        c = c.reshape(-1, 1)
    if len(n.shape) < 2:
        n = n.reshape(-1, 1)
    return np.hstack([c, n])


class DataFrameMapper:
    def __init__(self, nominal_columns: Optional[List[str]] = None,
                 one_hot_columns: Optional[List[Tuple[int, int]]] = None):

        if nominal_columns is None:
            self._nominal_columns = []
        else:
            self._nominal_columns = nominal_columns

        if one_hot_columns is None:
            self._one_hot_columns = []
        else:
            self._one_hot_columns = one_hot_columns

        self._continuous_columns: List[str] 
        self._nominal_columns_original_positions: List[int] 

        self._assert_not_nominal_and_one_hot()
        self._one_hot_encoder: Optional[OneHotEncoder] = None
        self._standard_scaler: Optional[StandardScaler] = None
        self._original_columns: List[str]
        self._n_continuous_columns: int

    def _assert_not_nominal_and_one_hot(self):
        if self._one_hot_columns and self._nominal_columns:
            raise ValueError(f"You passed {ValueNominal.__name__} and {OneHot.__name__} in constraints, but only one "
                             f"of them should be used")

    @property
    def nominal_columns(self):
        return self._nominal_columns

    @property
    def one_hot_spans(self) -> List[Tuple[int, int]]:
        if self._one_hot_columns:
            return self._one_hot_columns

        if self._one_hot_encoder is None:
            return []

        spans = []
        one_hot_start = self._n_continuous_columns
        for category in self._one_hot_encoder.categories_:
            n_categories = len(category)
            spans.append((one_hot_start, one_hot_start + n_categories))
            one_hot_start += n_categories
        return spans

    @property
    def _n_one_hot_columns(self):
        if self._one_hot_encoder is None:
            return None
        return sum(len(category) for category in self._one_hot_encoder.categories_)

    def transformed_column_span(self, column: str) -> Tuple[int, int]:
        for col, span in zip(self._nominal_columns, self.one_hot_spans):
            if col == column:
                return span

        # column is continuous and is at the beginning of transformed array
        idx = self._continuous_columns.index(column)
        return idx, idx + 1

    def inverse_transform(self, x: NDArray[np.float32]) -> pd.DataFrame:

        sc = self._standard_scaler
        ohe = self._one_hot_encoder

        if ohe is None and sc is None:
            raise ValueError("There are no transformers fitted. Did you forgot to call fit()?")
        elif ohe is None and sc is not None:  # to make mypy happy...
            reconstructed_continuous = sc.inverse_transform(x)
            return pd.DataFrame(data=reconstructed_continuous, columns=self._original_columns)
        elif sc is None and ohe is not None:  # to make mypy happy
            reconstructed_nominal = ohe.inverse_transform(x)
            return pd.DataFrame(data=reconstructed_nominal, columns=self._original_columns)

        assert sc is not None and ohe is not None  # again, mypy...

        n_one_hot = self._n_one_hot_columns
        one_hot_columns = x[:, -n_one_hot:]
        continuous_columns = x[:, :-n_one_hot]

        reconstructed_continuous = sc.inverse_transform(continuous_columns)
        reconstructed_labels = ohe.inverse_transform(one_hot_columns)

        reconstructed = reconstructed_continuous.astype(object)
        for i, column in enumerate(sorted(self._nominal_columns_original_positions)):
            reconstructed = np.insert(reconstructed, column, reconstructed_labels[:, i], axis=1)

        return pd.DataFrame(data=reconstructed, columns=self._original_columns)

    def fit(self, x: pd.DataFrame, y=None, **fit_params):
        self._original_columns = list(x.columns)
        self._fit_nominal(x)
        self._fit_continuous(x)
        return self

    def fit_transform(self, x: pd.DataFrame) -> NDArray[np.float32]:
        self.fit(x)
        return self.transform(x)

    def transform(self, x: pd.DataFrame, y=None) -> NDArray[np.float32]:
        continuous = self._transform_continuous(x) if self._standard_scaler is not None else None
        nominal = self._transform_nominal(x) if self._one_hot_encoder is not None else None
        if continuous is None and nominal is None:
            raise ValueError("There are no transformers fitted. Did you forgot to call fit()?")
        if continuous is None and nominal is not None:
            return nominal
        if nominal is None and continuous is not None:
            return continuous
        assert continuous is not None and nominal is not None
        return _stack_continuous_nominal(continuous, nominal)

    def _fit_continuous(self, x: pd.DataFrame):
        x = x.drop(columns=self._nominal_columns)
        self._continuous_columns = x.columns.tolist()
        print(f'Continous columns: {self._continuous_columns}')
        self._n_continuous_columns = x.shape[1]
        if not self._continuous_columns:
            return
        sc = StandardScaler()
        sc.fit(x)
        self._standard_scaler = sc

    def _transform_continuous(self, x: pd.DataFrame) -> NDArray[np.float32]:
        x = x.drop(columns=self._nominal_columns)
        assert self._standard_scaler is not None
        x_numpy = self._standard_scaler.transform(x)
        assert isinstance(x_numpy, np.ndarray)
        return x_numpy

    def _fit_nominal(self, x: pd.DataFrame):
        if not self._nominal_columns or self._one_hot_columns:
            return
        nominal_columns_original_positions = []
        for i, column in enumerate(x.columns):
            if column in self._nominal_columns:
                nominal_columns_original_positions.append(i)

        nominal_columns = x[self._nominal_columns]

        ohe = OneHotEncoder(sparse=False)
        ohe.fit(nominal_columns)
        self._one_hot_encoder = ohe
        self._nominal_columns_original_positions = nominal_columns_original_positions

    def _transform_nominal(self, x: pd.DataFrame) -> NDArray[np.float32]:
        if self._one_hot_columns:
            return np.hstack([x.iloc[:, start:end + 1] for (start, end) in self._one_hot_columns])

        nominal_columns = x[self._nominal_columns]
        assert self._one_hot_encoder is not None
        one_hot_encoded = np.round(self._one_hot_encoder.transform(nominal_columns))
        assert isinstance(one_hot_encoded, np.ndarray)
        return one_hot_encoded
