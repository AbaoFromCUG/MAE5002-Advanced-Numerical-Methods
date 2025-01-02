import numpy as np


f = np.cos
a = -np.pi / 6
b = np.pi / 6

# f_1 = -np.sin
# f_2 = -np.cos
# f_3 = np.sin
f_4 = np.cos
max_f_4 = np.cos(0)
min_f_4 = np.cos(a)
# print(max_f_4)
# print(min_f_4)


def error(a, b, max_f_4, M):
    h = (b - a) / (2 * M)
    return np.abs((b - a) * max_f_4 * h**4 / 180)


for i in range(1, 20):
    e = error(a, b, max_f_4, i)
    print(f"M={i}, Error={e}")
