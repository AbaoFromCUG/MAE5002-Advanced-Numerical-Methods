import numpy as np
import math
from functools import reduce
import sympy as sp


def f(x):
    if isinstance(x, sp.Symbol):
        return sp.cos(x)
    return np.cos(math.pi * x)


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


newton(f, np.array([0, np.pi / 2, np.pi]))


def E(f, X):
    N = len(X)
    x = sp.Symbol("x")
    c = sp.Symbol("c")
    Y = f(X)

    Enx = (
        reduce(lambda a, b: a * b, [x - xi for xi in X])
        * sp.diff(f(c), c, N)
        / math.factorial(N)
    )
    print(f"E_{N-1}(x)=", Enx.expand())
    print(f"E_{N-1}(x)<=", sp.maximum(Enx.subs(c, np.pi / 2), x, sp.Interval(0, np.pi)))


E(f, np.array([0, np.pi / 2, np.pi]))
