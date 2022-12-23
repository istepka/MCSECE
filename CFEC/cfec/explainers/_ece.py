from typing import List

import numpy as np
import itertools
import sklearn.neighbors
import functools
import joblib
import psutil
import warnings
import pandas as pd
from numpy.typing import NDArray

from ..base import BaseExplainer
from ._cadex_parallel import compute_criterion  # type: ignore


class ECE(BaseExplainer):
    def __init__(self, k: int, columns: List[str], bces: List[BaseExplainer], dist: int, h: int,
                 lambda_: float, n_jobs=None):
        self._col_names = columns
        self.k = k
        self.bces = bces
        self.norm = dist
        self.h = h
        self.lambda_ = np.float32(lambda_)
        if n_jobs is None:
            self.n_jobs = psutil.cpu_count(logical=False)
        else:
            self.n_jobs = n_jobs
        self._cfs_len: int
        self._aggregated_cfs: NDArray[np.float32]

    def _aggregate_cfs(self, x) -> NDArray[np.float32]:
        list_cfs: List[NDArray[np.float32]] = []
        for bce in self.bces:
            bce_result: NDArray[np.float32] = np.asarray(bce.generate(x).values)
            for bce_r in bce_result:
                list_cfs.append(bce_r)
        cfs: NDArray[np.float32] = np.unique(np.asarray(list_cfs), axis=0)
        self._cfs_len = cfs.shape[0]
        assert isinstance(cfs, np.ndarray)
        return cfs

    def _choose_best_k(self, valid_cfs: NDArray[np.float32], x_series):
        x = x_series.values
        norms: NDArray[np.float32] = np.apply_along_axis(functools.partial(np.linalg.norm, ord=self.norm),
                                                         0, valid_cfs)
        C = list(valid_cfs / norms)
        k = min(self.k, self._cfs_len)
        if k != self.k:
            warnings.warn(f'k parameter > number of aggregated counterfactuals. Changing k from {self.k} to {k}',
                          UserWarning, stacklevel=3)
        if self._cfs_len <= self.h:
            warnings.warn(
                f"knn's h parameter >= number of aggregated counterfactuals. Changing h from  {self.h} to {self._cfs_len - 1}",
                UserWarning, stacklevel=3)
            self.h = self._cfs_len - 1
        k_subsets = list()
        for i in range(k):
            k_subsets += list(itertools.combinations(C, r=i + 1))
        knn_c = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
        c_np: NDArray[np.float32] = np.asarray(C)
        knn_c.fit(c_np, np.ones(shape=c_np.shape[0]))

        S_ids = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(compute_criterion)(knn_c, self.norm, self.lambda_, c_np, x, S) for S in k_subsets)
        selected = norms * k_subsets[np.argmax(np.asarray(S_ids))]
        return selected

    def generate(self, x: pd.Series) -> pd.DataFrame:
        self._aggregated_cfs = self._aggregate_cfs(x)
        k_subset = self._choose_best_k(self._aggregated_cfs, x)
        return pd.DataFrame(k_subset, columns=self._col_names)

    def get_aggregated_len(self):
        if self._cfs_len is None:
            raise AttributeError('Aggregation has not been performed yet')
        return self._cfs_len

    def get_aggregated_cfs(self):
        if self._aggregated_cfs is None:
            raise AttributeError('Aggregation has not been performed yet')
        return pd.DataFrame(self._aggregated_cfs, columns=self._col_names)
