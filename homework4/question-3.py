import numpy as np
import sympy as sp
from functools import reduce


def f(x):
    return x**x


print(f"f(1)={f(1)}")
print(f"f(1.25)={f(1.25)}")
print(f"f(1.5)={f(1.5)}")


x = sp.Symbol("x")
x_i = [1, 1.25, 1.5]
y_i = list(map(f, x_i))


def P_N(X, Y):
    N = len(X)
    Pn = reduce(
        lambda a, b: a + b,
        [
            Y[i]
            * reduce(
                lambda a, b: a * b,
                [(x - X[j]) / (X[i] - X[j]) for j in range(N) if j != i],
            )
            for i in range(N)
        ],
    )

    Pn = Pn.expand().collect(x)
    print("拉格朗日P_2(x)=", Pn)


P_N(x_i, y_i)


# 验证
# def pp(x):
#     return 1.54951318788779 * x**2 - 2.1995483555447 * x + 1.65003516765691
#
#
# print(pp(1))
# print(pp(1.25))
# print(pp(1.5))


def IP(x):
    return (
        1.54951318788779 * x**3 / 3 - 2.1995483555447 * x**2 / 2 + 1.65003516765691 * x
    )


print("均值:", (IP(1.5) - IP(1)) / (1.5 - 1))


fdx3 = sp.diff(f(x), x, 3)
print("f(x)的三阶导数: ", fdx3)
# max_fdx3 = sp.maximum(fdx3, x, sp.Interval(1, 1.5))
# print("f(x)的三阶导数在[1, 1.5]上的最大值： ", max_fdx3)
fdx4 = sp.diff(f(x), x, 4)
print("f(x)的四阶导数: ", fdx4)


def fff(x):
    return x**x * ((np.log(x) + 1) ** 3 + 3 * (np.log(x) + 1) / x - 1 / x**2)


print("M_3", fff(1.5))


print("E_2(x)", (0.25**3 * fff(1.5)) / (9 * np.sqrt(3)))
