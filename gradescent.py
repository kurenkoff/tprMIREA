import numpy as np


class GradientDescent:
    def __init__(self):
        pass

    def minimize(self, f, df, x, h, eps=0.1, max_iterations=1000):
        xk = np.array(x)
        xk.astype('float')

        for k in range(max_iterations):
            dfxk = df(xk)
            if np.sqrt(np.sum(np.square(dfxk))) <= eps:
                return xk
            xk = self.iteration(xk, f, df, h)
            print("x[{0}] = {1}".format(k,xk))
        return xk

    def iteration(self, xk, f, df, h, eps = 0.1):
        xk1 = xk - h * df(xk)
        if f(xk1) >= f(xk):
            h = h / 2
            return self.iteration(xk, f, df, h)
        return xk1


def tf(x):
    return x[0] ** 2 + x[0] * x[1] + x[1] ** 2 - 6 * x[0] - 9 * x[1]


def dtf(x):
    return np.array([2 * x[0] + x[1] - 6, x[0] + 2 * x[1] - 9])


if __name__ == "__main__":
    print("Метод градиентного спуска:")
    gd = GradientDescent()
    res = gd.minimize(tf, dtf, np.array([0., 0.]), 0.01)
    print("x = {0}, f(x) = {1}".format(res, tf(res)))