import numpy as np


def F(i):
    a, b = 0, 1
    while i > 0:
        a, b = b, a + b
        i = i - 1
    return a


# def fibonacci_search(f, df, a, b):
#     gr = (np.sqrt(5) - 1) / 2
#     for i in range(3):
#         print(f"a_{i}={a}, b_{i}={b}")
#         c_i = a + (1 - F(n - k - 1) / F(n - k)) * (b - a)
#         d_i = a * (1 - gr) + gr * b
#         if f(c_i) < f(d_i):
#             a, b = a, d_i
#         else:
#             a, b = c_i, b

i = 1
a, b = 0, 1
while True:
    a, b = b, a + b
    if 10**-8 > (3.99 - 3.33) / a:
        print(i, a)
        break
    i = i + 1
