from functools import reduce
import numpy as np
import sympy as sp


def L_i(X):
    x = sp.Symbol("x")
    N = len(X)
    Li = [
        reduce(
            lambda a, b: a * b,
            [(x - X[j]) / (X[i] - X[j]) for j in range(N) if j != i],
        )
        for i in range(N)
    ]
    for i in range(N):
        Li = reduce(
            lambda a, b: a * b,
            [(x - X[j]) / (X[i] - X[j]) for j in range(N) if j != i],
        )
        print(f"L_{i}=", Li.expand())


L_i(
    [
        sp.cos(5 * sp.pi / 6),
        0,
        sp.cos(sp.pi / 6),
    ]
)
