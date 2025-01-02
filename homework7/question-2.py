import numpy as np


def f(x):
    return 2 * np.pi * np.sin(x) * np.sqrt(1 + (np.cos(x)) ** 2)


def trapez(a, b, M, f):
    h = (b - a) / M
    steps = np.linspace(a, b, M + 1, endpoint=True)
    y = np.vectorize(f)(steps)
    return h / 2 * (y[0] + y[-1] + 2 * np.sum(y[1:-1]))


print(trapez(0, np.pi / 4, 10, f))


def simpson(a, b, M, f):
    h = (b - a) / (2 * M)
    steps = np.linspace(a, b, 2 * M + 1, endpoint=True)
    y = np.vectorize(f)(steps)
    w = np.ones_like(y)
    w[1:-1:2] = 4
    w[2:-2:2] = 2
    return h / 3 * np.sum(w * y)

print(simpson(0, np.pi / 4, 5, f))



