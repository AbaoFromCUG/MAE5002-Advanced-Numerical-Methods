import numpy as np
import matplotlib.pyplot as plt

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


##########################################
# 作业五 题目五（本题）的代码：给定条件求三次样条曲线参数
##########################################


def solve_cubic_coeff_extra(points: np.ndarray):
    """给定点，解三次样条曲线系数，方法为如下等式的线性方程组Ac=b
    f_i(i, x) = y,            (x, y) 为point
    fd_i(i, x) = fd_i(i, x)， (x, y) 为point（不包括头尾
    fdd_i(i, x) = fdd_i(i, x)， (x, y) 为第i个point（不包括头尾

    Args:
        point: (N, 2) 所有的插值点
        start_der: 起始点一阶导数值
        end_der: 结束点一阶导数值
    """
    point_num = len(points)
    interval_num = point_num - 1
    coeff_num = (point_num - 1) * 4

    def f_i(i, x):
        """f_i(x) = y 对应的系数，即Ac =b的某一行

        Args:
            i (int): 第i条曲线
            x (float): x值
        """
        row = np.zeros((coeff_num))
        start = i * 4
        row[start : start + 4] = [1, x, x**2, x**3]
        return row

    def fd_i(i, x):
        """一阶倒数系数"""
        row = np.zeros(coeff_num)
        start = i * 4
        row[start : start + 4] = [0, 1, 2 * x, 3 * x**2]
        return row

    def fdd_i(i, x):
        """二阶倒数系数"""
        row = np.zeros(coeff_num)
        start = i * 4
        row[start : start + 4] = [0, 0, 2, 6 * x]
        return row

    A = []
    b = []
    for i, (x, y) in enumerate(points):
        if i != point_num - 1:
            print(f"point{i} 通过 spine{i}")
            A.append(f_i(i, x))
            b.append(y)
        if i != 0:
            print(f"point{i} 通过 spine{i-1}")
            A.append(f_i(i - 1, x))
            b.append(y)
    for i in range(1, point_num - 1):
        (x, y) = points[i]
        print(f"point{i} spine{i-1} 和 spine{i} 一阶导数连续（相等）")
        A.append(fd_i(i - 1, x) - fd_i(i, x))
        b.append(0)
        print(f"point{i} spine{i-1} 和 spine{i} 二阶导数连续（相等）")
        A.append(fdd_i(i - 1, x) - fdd_i(i, x))
        b.append(0)
    # 最后边界条件
    A.append(fdd_i(0, points[0, 0]))
    b.append((points[1, 1] - 2 * points[0, 1]) / ((points[1, 0] - points[0, 0]) ** 2))
    A.append(fdd_i(interval_num - 1, points[-1, 0]))
    b.append(
        (points[-2, 1] - 2 * points[-1, 1]) / ((points[-2, 0] - points[-1, 0]) ** 2)
    )
    A = np.array(A)
    # print(A)
    B = np.array(b).reshape([-1, 1])
    C = gs_partial_scaled_pivot(A, B)

    def S(x):
        if x < points[0, 0] and x > points[-1, 0]:
            assert False, "out of range"
        for i in range(interval_num):
            if x <= points[i + 1, 0]:
                return np.sum(C[i * 4 : i * 4 + 4] @ [1, x, x**2, x**3])

    print(C)
    for i in range(interval_num):
        start = i * 4

        p1, p2, p3, p4 = C[start : start + 4]
        print(f"{p1} + {p2} x + {p3} x^2 + {p4} x^3")
    return np.vectorize(S)


p = np.array(
    [
        [0, 0],
        [2, 40],
        [4, 160],
        [6, 300],
        [8, 480],
    ]
)
model = solve_cubic_coeff_extra(p)

x_vals = np.linspace(0, 8, 100)

plt.plot(x_vals, model(x_vals), label="Time-Distance")
plt.scatter(p[:, 0], p[:, 1], color="red", label="data points")
plt.legend()
plt.show()
