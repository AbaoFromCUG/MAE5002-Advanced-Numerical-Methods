import numpy as np


##########################################
# copy from 作业四 的题目四的代码：解线性方程组
##########################################


def back_substitution(AB: np.ndarray):
    M, N = AB.shape
    assert M + 1 == N
    X = np.zeros(M)
    for i in reversed(range(M)):
        X[i] = (AB[i, -1] - np.sum(AB[i, :-1] * X)) / AB[i, i]

    return X


def gs_partial_pivot(A: np.ndarray, B: np.ndarray):
    M, M = A.shape
    AB = np.hstack((A, B))

    print("conbine augmented matrix AB:")
    # print(AB)

    # 消元
    for p in range(M - 1):
        # print(f"check pivoting p={p}")

        k = np.argmax(np.abs(AB[p:, p])) + p
        if k > p:
            print(f"pivoting swap {p} and {k}")
            AB[[p, k], :] = AB[[k, p], :]

        for i in range(p + 1, M):
            AB[i, :] = AB[i, :] - AB[p, :] * AB[i, p] / AB[p, p]
        # print(f"elimination pivoting {p}")
        # print(AB)

    return back_substitution(AB)


def gs_partial_scaled_pivot(A: np.ndarray, B: np.ndarray):
    M, M = A.shape
    AB = np.hstack((A, B))

    # print("conbine augmented matrix AB:")
    # print(AB)

    # 消元
    for p in range(M - 1):
        # print(f"check scaled partical pivoting p={p}")

        k = np.argmax(np.abs(AB[p:, p] / np.max(np.abs(AB[p:, p:]), axis=1))) + p
        if k > p:
            # print(f"pivoting swap {p} and {k}")
            AB[[p, k], :] = AB[[k, p], :]

        for i in range(p + 1, M):
            AB[i, :] = AB[i, :] - AB[p, :] * AB[i, p] / AB[p, p]
        # print(f"elimination pivoting {p}")
        # print(AB)

    return back_substitution(AB)


A = np.array(
    [
        [1, 1, 1, 1, 1],
        [0, 1, 2, 3, 4],
        [0, 1, 4, 9, 16],
        [0, 1, 8, 27, 64],
        [0, 1, 16, 81, 256],
    ]
)
B = np.array([4, 8, 64 / 3, 64, 1024 / 5]).reshape([-1, 1])

C = gs_partial_scaled_pivot(A, B)
print(C)
for i, w in enumerate(C):
    print(f"w_{i}={w}")
