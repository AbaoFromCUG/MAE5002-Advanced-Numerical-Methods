import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def quadratic_approx(f, a, b, tol=1e-5, max_iter=10000):
    # print(a, b)
    p_0, p_1, p_2, h = a, (a + b) / 2, b, (b - a) / 2
    for i in range(max_iter):
        y_0, y_1, y_2 = f(p_0), f(p_1), f(p_2)
        h = h * (4 * y_1 - 3 * y_0 - y_2) / (4 * y_1 - 2 * y_0 - 2 * y_2)
        p_0 = p_0 + h
        p_1 = p_0 + h
        p_2 = p_0 + h * 2
        # print(p_0, p_1, p_2, h)
        if abs(h * 2) < tol:
            break
    return p_1


def modified_newtons(f, df, hf, p, tol=1e-5, max_iter=10):
    X, Y, Z = [p[0]], [p[1]], [f(p)]
    for i in range(max_iter):
        grad = df(p)
        hessian = hf(p)
        step = -np.matmul(grad, np.linalg.inv(hessian).T)
        print(hessian.shape)
        print(grad.shape)
        print(step.shape)

        def fn(g):
            return f(p + g * step)

        gamma = quadratic_approx(fn, 0, 100)
        new_p = p + gamma * step

        if np.abs(f(p) - f(new_p)) < tol:
            break
        p = new_p
        print(f"z={f(p)}, gamma = {gamma}")
        X.append(p[0])
        Y.append(p[1])
        Z.append(f(p))
    return np.array(X), np.array(Y), np.array(Z)


def f(p):
    x, y = p[0], p[1]
    # print("---",x, y, p)
    return x * x + y**3 - 3 * x - 3 * y + 5


def pf(p):
    x, y = p
    return np.array([2 * x - 3, 3 * y * y - 3])


def hf(p):
    x, y = p
    return np.array(
        [
            [2, 0],
            [2, 6 * y],
        ]
    )


step_X, step_Y, step_Z = modified_newtons(f, pf, hf, [-1, 2])
print(step_Z)


X, Y = np.mgrid[-5:5:0.1, -5:5:0.1]
Z = f(np.stack([X, Y], axis=0))

ax = plt.figure().add_subplot(projection="3d")

ax.plot(step_X, step_Y, step_Z, label="Step curve")
ax.scatter(-1, 2, f([-1, 2]), label="Start point")
ax.scatter(step_X, step_Y, step_Z, label="Step points")
print(step_X, step_Y, step_Z)

# Plot the 3D surface
ax.plot_surface(X, Y, Z, edgecolor="royalblue", lw=0.5, rstride=8, cstride=8, alpha=0.3)

ax.contourf(X, Y, Z, zdir="z", offset=-100, cmap="coolwarm")
# ax.contourf(X, Y, Z, zdir="x", offset=-40, cmap="coolwarm")
# ax.contourf(X, Y, Z, zdir="y", offset=40, cmap="coolwarm")

ax.set(xlim=(-5, 5), ylim=(-5, 5), zlim=(-100, 100), xlabel="X", ylabel="Y", zlabel="Z")
ax.legend()
plt.show()
