from functools import reduce
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def fit(f, X):
    N = len(X)
    Y = f(X)
    x = sp.Symbol("x")
    Pn = reduce(
        lambda a, b: a + b,
        [
            Y[i]
            * reduce(
                lambda a, b: a * b,
                [(x - X[j]) / (X[i] - X[j]) for j in range(N) if j != i],
            )
            for i in range(N)
        ],
    )

    def inference(val):
        return Pn.subs(x, val)

    return np.vectorize(inference)


input = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])


plot_x = np.linspace(-1, 2, 10, endpoint=True)

f = np.exp
plt.plot(plot_x, fit(f, input)(plot_x), label="P(x)")
plt.plot(plot_x, f(plot_x), label="f(x)")
plt.scatter(input, f(input))
plt.legend()
plt.savefig("figure-exp.png")
plt.show()

f = np.sin
plt.plot(plot_x, fit(f, input)(plot_x), label="P(x)")
plt.plot(plot_x, f(plot_x), label="f(x)")
plt.scatter(input, f(input))
plt.legend()
plt.savefig("figure-sin")
plt.show()


f = np.vectorize(lambda x: (x + 1) ** (x + 1))
plt.plot(plot_x, fit(f, input)(plot_x), label="P(x)")
plt.plot(plot_x, f(plot_x), label="f(x)")
plt.scatter(input, f(input))
plt.legend()
plt.savefig("figure-x+1")
plt.show()

