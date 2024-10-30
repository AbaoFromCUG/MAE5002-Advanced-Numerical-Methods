import numpy as np

np.set_printoptions(suppress=True)


# %%


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
    print(AB)

    # 消元
    for p in range(M - 1):
        print(f"check pivoting p={p}")

        k = np.argmax(np.abs(AB[p:, p])) + p
        if k > p:
            print(f"pivoting swap {p} and {k}")
            AB[[p, k], :] = AB[[k, p], :]

        for i in range(p + 1, M):
            AB[i, :] = AB[i, :] - AB[p, :] * AB[i, p] / AB[p, p]
        print(f"elimination pivoting {p}")
        print(AB)

    return back_substitution(AB)


def gs_partial_scaled_pivot(A: np.ndarray, B: np.ndarray):
    M, M = A.shape
    AB = np.hstack((A, B))

    print("conbine augmented matrix AB:")
    print(AB)

    # 消元
    for p in range(M - 1):
        print(f"check scaled partical pivoting p={p}")

        k = np.argmax(np.abs(AB[p:, p] / np.max(np.abs(AB[p:, p:]), axis=1))) + p
        if k > p:
            print(f"pivoting swap {p} and {k}")
            AB[[p, k], :] = AB[[k, p], :]

        for i in range(p + 1, M):
            AB[i, :] = AB[i, :] - AB[p, :] * AB[i, p] / AB[p, p]
        print(f"elimination pivoting {p}")
        print(AB)

    return back_substitution(AB)


# %%
A = np.array([[2, -3, 100], [1, 10, -0.001], [3, -100, 0.01]])
B = np.array([1, 0, 0]).reshape([-1, 1])

# %%
print(gs_partial_pivot(A, B))

# %%
print(gs_partial_scaled_pivot(A, B))


# %%
A = np.array(
    [[1, 20, -1, 0.001], [2, -5, 30, -0.1], [5, -1, -100, -10], [2, -100, -1, -1]]
)
B = np.array([0, 1, 0, 0]).reshape([-1, 1])

# %%
print(gs_partial_pivot(A, B))

# %%
print(gs_partial_scaled_pivot(A, B))
