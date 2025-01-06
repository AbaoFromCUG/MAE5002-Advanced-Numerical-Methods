import numpy as np


def quadratic_approx(f, a, b):
    p_0, p_1, p_2, h = a, (a + b) / 2, b, (b - a) / 2
    for i in range(3):
        print(f"iteration i={i}, p_0={p_0}, p_1={p_1}, p_2={p_2}, h={h}")
        y_0, y_1, y_2 = f(p_0), f(p_1), f(p_2)
        h = h * (4 * y_1 - 3 * y_0 - y_2) / (4 * y_1 - 2 * y_0 - 2 * y_2)
        p_0 = p_min = p_0 + h
        p_1 = p_0 + h
        p_2 = p_0 + h * 2

quadratic_approx(lambda x: -np.sin(x) - x + x * x / 2, 0.8, 1.6)
