import sympy as sp
import numpy as np


def calculate(values):
    """
    AX=B
    """
    A = np.zeros((3, 3))
    A[:, 0] = 1
    A[:, 1:] = values[:, :2]
    print("系数矩阵：")
    print(A)
    print("y：", values[:, 2])
    X = np.linalg.inv(A).dot(values[:, 2])
    print("解为：", X)


calculate(
    np.array(
        [
            [1, 1, 5],
            [2, 1, 3],
            [1, 2, 9],
        ]
    )
)

calculate(
    np.array(
        [
            [1, 1, 2.5],
            [2, 1, 0],
            [1, 2, 4],
        ]
    )
)
calculate(
    np.array(
        [
            [2, 1, 5],
            [1, 3, 7],
            [3, 2, 4],
        ]
    )
)

calculate(
    np.array(
        [
            [1, 2, 5],
            [3, 2, 7],
            [1, 2, 0],
        ]
    )
)
