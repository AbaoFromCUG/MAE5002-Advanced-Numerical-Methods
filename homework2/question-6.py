from math import log, exp


def bisect(f, a: float, b: float, delta: float):
    ya = f(a)
    yb = f(b)

    if ya * yb > 0:
        return
    max = 1 + int(round((log(b - a) - log(delta)) / log(2)))
    for _ in range(1, max):
        c = (a + b) / 2
        yc = f(c)
        if yc == 0:
            a = c
            b = c
        elif yb * yc > 0:
            b = c
            yb = yc
        else:
            a = c
            ya = yc
        if b - a < delta:
            break
    c = (a + b) / 2
    err = abs(b - a)
    yc = f(c)
    return c, err, yc


def regula(f, a: float, b: float, delta: float, epsilon: float, max: int):
    ya = f(a)
    yb = f(b)
    if ya * yb > 0:
        print("Note: f(a)*f(b)>0")
        return
    for _ in range(1, max):
        dx = yb * (b - a) / (yb - ya)
        c = b - dx
        ac = c - 1
        yc = f(c)
        if yc == 0:
            break
        if yb * yc > 0:
            b = c
            yb = yc
        else:
            a = c
            ya = yc
        dx = min(abs(dx), ac)
        if abs(dx) < delta:
            # print("delta")
            break
        if abs(yc) < epsilon:
            # print("epsilon")
            break
    err = abs(b - a) / 2
    yc = f(c)
    return c, err, yc


def f(x):
    return x * x - exp(x)


print(bisect(f, -2, 2, 0.00001))
print(regula(f, -2, 2, 0.00001, 0.00001, 100000))
