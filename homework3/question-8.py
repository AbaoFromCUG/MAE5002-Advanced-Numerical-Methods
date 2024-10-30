import numpy as np


N = 50
A = np.zeros((N, N))
for j in range(N):
    A[j][j] = 12
    if j > 0:
        A[j][j - 1] = -2
    if j > 1:
        A[j][j - 2] = 1

    if j < N - 1:
        A[j][j + 1] = -2
    if j < N - 2:
        A[j][j + 2] = 1


B = np.ones(N) * 5

# %%
# Gauss-Seidel iteration

# assert len(A.shape) == 2
# assert A.shape[0] == A.shape[1]
# assert len(B.shape) == 1
# assert A.shape[1] == B.shape[0]


def gs_iteration(A: np.ndarray, B: np.ndarray, max_iter=10000, max_residual=0.0001):
    X = np.zeros(N)  # 初始化X_0
    # 最多迭代max_iter次
    for i in range(max_iter):
        X = X.copy()  # 拷贝一份Xi
        for j in range(N):
            # 顺序更新xi (即每个未知数)
            X[j] = (
                B[j] - np.sum(A[j, :j] * X[:j]) - np.sum(A[j, j + 1 :] * X[j + 1 :])
            ) / A[j, j]
        # 所有的未知数都更新了一遍，计算误差（2-norm）
        residual = np.linalg.norm(A @ X - B)
        # 当误差小于要求值
        if max_residual > residual:
            return X

    return X


# %%
gs_iteration(A, B)
