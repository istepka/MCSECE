# type: ignore

import numpy as np


def compute_criterion(knn_c, norm, lambda_, C, x, S):
    bin_sumset = np.full(shape=len(S), fill_value=False, dtype=bool)
    dist_sum = 0.
    S_np = np.asarray(S)
    C = np.asarray(C)
    for c in S:
        (_, ids) = knn_c.kneighbors(np.expand_dims(c, axis=0))
        neighbors = np.squeeze(C[ids[:, 1:]])
        for i in range(neighbors.shape[0]):
            bin_sumset |= np.all(neighbors[i] == S_np, axis=1)
        minuend = np.sum(bin_sumset)
        dist_sum += np.linalg.norm(x - c, ord=norm)
    subtrahend = lambda_ * dist_sum
    return minuend - subtrahend
