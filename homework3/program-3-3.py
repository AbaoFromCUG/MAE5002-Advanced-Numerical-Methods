import numpy as np


def lufact(A: np.ndarray, B: np.ndarray):
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert len(B.shape) == 2
    assert A.shape[1] == B.shape[1]
    assert B.shape[1] == 1

    N = A.shape[0]
    X = np.zeros((N, 1))
    Y = np.zeros((N, 1))
    C = np.zeros((1, N))

    return X
