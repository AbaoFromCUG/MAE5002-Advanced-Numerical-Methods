import numpy as np


def golden_ratio_search(f, df, a, b):
    gr = (np.sqrt(5) - 1) / 2
    for i in range(3):
        print(f"a_{i}={a}, b_{i}={b}")
        c_i = a * gr + (1 - gr) * b
        d_i = a * (1 - gr) + gr * b
        if f(c_i) < f(d_i):
            a, b = a, d_i
        else:
            a, b = c_i, b


def f(x):
    return np.exp(x) + 2 * x + x * x / 2


def df(x):
    return np.exp(x) + 2 + x


golden_ratio_search(f, df, -2.4, -1.6)
