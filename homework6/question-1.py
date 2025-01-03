import numpy as np


def f_x(f, x, y, h):
    return (f(x + h, y) - f(x - h, y)) / (2 * h)


def f_y(f, x, y, h):
    return (f(x, y + h) - f(x, y - h)) / (2 * h)


print("---------------")
print("(a) part:")


def f(x, y):
    return x * y / (x + y)


for h in [0.1, 0.01, 0.001]:
    print(f"while h={h}:")
    print(f"f_x(2, 3)={f_x(f, 2,3, h)}")
    print(f"f_y(2, 3)={f_y(f, 2,3, h)}")

print("---------------")
print("(b) part:")


def z(x, y):
    return np.arctan(y / x)


for h in [0.1, 0.01, 0.001]:
    print(f"while h={h}:")
    print(f"f_x(3, 4)={f_x(z, 3, 4, h)}")
    print(f"f_y(3, 4)={f_y(z, 3, 4, h)}")
