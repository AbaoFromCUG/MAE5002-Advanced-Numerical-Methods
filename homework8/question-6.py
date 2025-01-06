from enum import global_enum
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def quadratic_approx(f, a, b, tol=1e-5, max_iter=10000):
    p_0, p_1, p_2, h = a, (a + b) / 2, b, (b - a) / 2
    for i in range(max_iter):
        y_0, y_1, y_2 = f(p_0), f(p_1), f(p_2)
        h = h * (4 * y_1 - 3 * y_0 - y_2) / (4 * y_1 - 2 * y_0 - 2 * y_2)
        p_0 = p_0 + h
        p_1 = p_0 + h
        p_2 = p_0 + h * 2
        if abs(h * 2) < tol:
            break
    return p_1


def gradient(f, pf, x, y, tol=1e-5, max_iter=30):
    z = f(x, y)
    X, Y, Z = [], [], []

    print(z)
    for i in range(max_iter):
        gx, gy = pf(x, y)
        norm = np.sqrt(gx**2 + gy**2)
        Sx, Sy = -gx / norm, -gy / norm

        def grid_fn(gamma):
            return f(x + gamma * Sx, y + gamma + Sy)

        gamma = quadratic_approx(grid_fn, 0, 10)
        new_x, new_y = x + gamma * Sx, y + gamma * Sy
        new_z = f(new_x, new_y)

        if np.abs(z - new_z) < tol:
            break
        x, y, z = new_x, new_y, new_z
        X.append(x)
        Y.append(y)
        Z.append(z)
        print(f"gamma={gamma}, P_{i+1}=({float(x)}, {float(y)})")
    return np.array(X), np.array(Y), np.array(Z)


def f(x, y):
    return x * x + y**3 - 3 * x - 3 * y + 5


def pf(x, y):
    return [2 * x - 3, 3 * y * y - 3]


step_X, step_Y, step_Z = gradient(f, pf, -1, 2)
print(step_Z)


X, Y = np.mgrid[-5:5:0.1, -5:5:0.1]
Z = np.vectorize(f)(X, Y)

ax = plt.figure().add_subplot(projection="3d")

ax.plot(step_X, step_Y, step_Z, label="Step curve")
ax.scatter(-1, 2, f(-1, 2), label="Start point")
ax.scatter(step_X, step_Y, step_Z, label="Step points")

# Plot the 3D surface
ax.plot_surface(X, Y, Z, edgecolor="royalblue", lw=0.5, rstride=8, cstride=8, alpha=0.3)

ax.contourf(X, Y, Z, zdir="z", offset=-100, cmap="coolwarm")
# ax.contourf(X, Y, Z, zdir="x", offset=-40, cmap="coolwarm")
# ax.contourf(X, Y, Z, zdir="y", offset=40, cmap="coolwarm")

ax.set(xlim=(-5, 5), ylim=(-5, 5), zlim=(-100, 100), xlabel="X", ylabel="Y", zlabel="Z")
ax.legend()
plt.show()
