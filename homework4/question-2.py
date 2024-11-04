def P(x):
    return -0.02 * x**3 + 0.1 * x**2 - 0.2 * x + 1.66


print(P(1))
print(P(2))
print(P(3))
print(P(4))
print(P(5))


def PP(x):
    return -0.06 * x**2 + 0.2 * x - 0.2


def P0(x):
    return 0.005 * (x**4) + 0.1 * x**3 / 3 - 0.1 * x**2 + 1.66 * x


print(PP(4))

print(P0(4) - P0(1))
print(f"P(5)={P(5)}")
