import numpy as np
from functools import reduce
import sympy as sp


def f(x):
    return np.exp(-x)


def newton(f, x):
    N = len(x)
    fx = f(x)
    print(fx)
    next_fx = fx
    f_f = []
    for i in range(2, N + 1):
        next_fx = np.array(
            [
                (next_fx[j + 1] - next_fx[j]) / (x[j + i - 1] - x[j])
                for j in range(N - i + 1)
            ]
        )
        f_f.append(next_fx)
        print(next_fx)

    X = sp.Symbol("x")
    Pi = fx[0]
    for i in range(N - 1):
        Pi = Pi + f_f[i][0] * reduce(
            lambda a, b: a * b, [(X - xi) for xi in x[: i + 1]]
        )
        print(f"P_{i+1}=", Pi.expand())


newton(f, np.array([0, 1, 2, 3, 4]))
print("增加x=0.5, 1.5之后")
newton(f, np.array([0, 1, 2, 3, 4, 0.5, 1.5]))
