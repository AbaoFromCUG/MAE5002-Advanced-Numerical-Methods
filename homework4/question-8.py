import sympy as sp
from functools import reduce


def R(f, m, n):
    a = [sp.Symbol(f"a{i}") for i in range(m + 1)]
    b = [sp.Symbol(f"b{j}") for j in range(1, n + 1)]

    x = sp.Symbol("x")
    Rx = sum([a[j] * x**j for j in range(m + 1)]) / (
        1 + sum([b[k] * x ** (k + 1) for k in range(0, n)])
    )

    print(Rx.subs(x, 0))
    print(sp.diff(Rx, x).subs(x, 0))
    print(sp.diff(Rx, x, 2).subs(x, 0))
    print(sp.diff(Rx, x, 3).subs(x, 0))
    print(sp.diff(Rx, x, 4).subs(x, 0))


def f(x):
    return sp.ln(1 + x) / x


[p0, p1, p2] = [sp.Symbol(f"p{i}") for i in range(3)]
[q1, q2] = [sp.Symbol(f"q{j}") for j in range(1, 3)]


x = sp.Symbol("x")

Zx = (1 - x / 2 + x**2 / 3 - x**3 / 4 + x**4 / 5) * (1 + q1 * x + q2 * x**2) - (
    p0 + p1 * x + p2 * x**2
)
print("解方程组：")
for i in range(5):
    print(Zx.expand().collect(x).coeff(x, i), "=0")
# print(
#     ([Zx.expand().collect(x).coeff(x, i) for i in range(4)][-2:]),
#     [p0, p1, p2, q1, q2][-2:],
# )
values = sp.solve(
    ([Zx.expand().collect(x).coeff(x, i) for i in range(5)]), [p0, p1, p2, q1, q2]
)
print(values)
