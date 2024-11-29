from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from numpy._core.defchararray import endswith


def E2(table, f):
    """计算均方根误差

    Args:
        table     [N, 2] 表示(xi, yi)
        f    fun: 函数
    Returns:
        均方根误差
    """
    X, Y = table[:, 0], table[:, 1]
    f = np.vectorize(f)
    return np.sqrt(np.sum(np.subtract(f(X), Y) ** 2) / len(X))


def plot(table, f):
    X, Y = table[:, 0], table[:, 1]
    x_min, x_max = np.min(X), np.max(X)
    f = np.vectorize(f)
    X_line = np.linspace(x_min, x_max, 1000, endpoint=True)
    plt.figure()
    plt.scatter(X, Y)
    plt.plot(X_line, f(X_line))
    plt.show()


tab1 = np.array(
    [
        [-1, 13.45],
        [0, 3.01],
        [1, 0.67],
        [2, 0.15],
    ]
)

tab2 = np.array(
    [
        [-1, 13.65],
        [0, 1.38],
        [1, 0.49],
        [3, 0.15],
    ]
)


def solve(X, Y) -> Tuple[float, float]:
    """使用最小二乘法进行线性拟合，得到A和B

    Args:
        X    [N]:   数组，N个元素
        Y    [N]:   数组，N个元素
    Returns:
        返回 (A, B) A 为斜率， B为截距
    """
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    C = np.sum((X - x_mean) ** 2)
    A = np.sum((X - x_mean) * (Y - y_mean)) / C
    B = y_mean - A * x_mean
    return A, B


def resolve(table, input_mapper, coeff_mapper):
    """
    将数据按照x_mapper，y_mapper后，返回原始系数

    Args:
        table     [N, 2]
        input_mapper (x_mapper, y_mapper) 数据mapper
            x_mapper (fun(x):X): 映射函数由x到X
            y_mapper (fun(y):Y): 隐射函数由y到Y
        coeff_mapper (A_mapper, B_mapper) 系数mapper
            A_mapper (fun(A):raw_A): 映射系数A回到原参数
            B_mapper (fun(B):raw_B): 隐射系数B回到原参数
    Returns:
        返回 (raw_A, raw_B) 原函数系数
    """
    x_mapper, y_mapper = input_mapper
    x_mapper, y_mapper = np.vectorize(x_mapper), np.vectorize(y_mapper)
    X, Y = table[:, 0], table[:, 1]
    A, B = solve(x_mapper(X), y_mapper(Y))
    A_mapper, B_mapper = coeff_mapper
    return A_mapper(A), B_mapper(B)


def do_nothing(x):
    return x


#################################
# (a) i
#################################
A, C = resolve(tab1, [do_nothing, np.log], [do_nothing, np.exp])


def f(x):
    return C * np.exp(A * x)


e2 = E2(tab1, f)
plot(tab1, f)

print("(a) i")
print(f" A={A}, C={C}, E2={e2}")

#################################
# (a) ii
#################################
A, C = resolve(tab2, [do_nothing, np.log], [do_nothing, np.exp])


def f(x):
    return C * np.exp(A * x)


e2 = E2(tab2, f)
plot(tab2, f)

print("(a) ii")
print(f" A={A}, C={C}, E2={e2}")


#################################
# (b) i
#################################
A, B = resolve(
    tab1, [do_nothing, lambda y: np.power(y, -0.5)], [do_nothing, do_nothing]
)


def f(x):
    return np.power(A * x + B, -2)


e2 = E2(tab1, f)
plot(tab1, f)

print("(b) i")
print(f" A={A}, B={B}, E2={e2}")

#################################
# (b) ii
#################################
A, B = resolve(
    tab2, [do_nothing, lambda y: np.power(y, -0.5)], [do_nothing, do_nothing]
)


def f(x):
    return np.power(A * x + B, -2)


e2 = E2(tab2, f)
plot(tab2, f)

print("(b) ii")
print(f" A={A}, B={B}, E2={e2}")
