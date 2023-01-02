import numpy as np
from numpy.typing import ArrayLike


def getL1distanceMatrix(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    if len(x.shape) == 1:
        return np.sum(np.abs(x - y))
    return np.sum(np.abs(x - y), axis=1)

def getL1distanceFloat(x: float, y: float) -> float:
    return abs(x - y)

def getL2distanceMatrix(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    if len(x.shape) == 1:
        return np.sum((x - y) ** 2)
    return np.sum((x - y) ** 2, axis=1)

def getL2distanceFloat(x: float, y: float) -> float:
    return (x - y) ** 2


if __name__ == '__main__':
    m1 = np.array([1,1,0,2])
    m2 = np.array([0,1,0,1.5])
    m3 = np.array([
        [1,2,3],
        [1,1,1]
    ])
    m4 = np.array([
        [1,0,0],
        [1,1,1]
    ])

    print(m1.shape)
    print(m3.shape)


    print(getL1distanceFloat(2,3))
    print(getL1distanceMatrix(m1, m2))
    print(getL1distanceMatrix(m3, m4))

    print(getL2distanceFloat(1,3))
    print(getL2distanceMatrix(m1, m2))
    print(getL2distanceMatrix(m3, m4))