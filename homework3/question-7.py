import numpy as np
import sympy as sp


# %%


# %%
x, A1, A2, A3, A4, A5, A6 = sp.symbols("x, A1, A2,A3, A4, A5, A6")

# left = (
#     (x - 2) * (x - 3) ** 2 * (x ** 2 + 1) * A1
#     + (x - 1) * (x - 3) ** 2 * (x ** 2 + 1) * A2
#     + (x - 1) * (x - 2) * (x ** 2 + 1) * A3
#     + (x - 1) * (x - 2) * (x - 3) * (x ** 2 + 1) * A4
#     + x * (x - 1) * (x - 2) * (x - 3) ** 2 * A5
#     + (x - 1) * (x - 2) * (x - 3) ** 2 * A6
# )

left = (x**2 + x + 1) / ((x - 1) * (x - 2) * (x - 3) ** 2 * (x**2 + 1))

right = (
    (1 / (x - 1)) * A1
    + (1 / (x - 2)) * A2
    + (1 / (x - 3) ** 2) * A3
    + A4 / (x - 3)
    + (A5 * x + A6) / (x**2 + 1)
)

# %%
left

# %%
right

# %%
# %% [md]
"""
# 两边同时乘以分母
"""
# %%


left = sp.simplify(left * ((x - 1) * (x - 2) * (x - 3) ** 2 * (x**2 + 1)))
right = sp.simplify(right * ((x - 1) * (x - 2) * (x - 3) ** 2 * (x**2 + 1)))

# %%

right = right.expand().collect(x)
right

# %%
left

# %%
right

# %%
A = [A1, A2, A3, A4, A5, A6]

b = [left.coeff(x, i) for i in reversed(range(6))]

# %%%

E = [right.coeff(x, i) for i in reversed(range(6))]

# %%
E

# %%
b

# %%

M = [[right.coeff(x, i).coeff(A[j]) for j in range(6)] for i in reversed(range(6))]

# %%
M


# %%
result = sp.solve([e - i for e, i in zip(E, b)], A)

# %%
print(result)
