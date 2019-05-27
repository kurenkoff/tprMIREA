import numpy as np


def tf(x):
    return x[0] ** 2 + x[0] * x[1] + x[1] ** 2 - 6 * x[0] - 9 * x[1]


def dtf(x):
    return np.array([2 * x[0] + x[1] - 6, x[0] + 2 * x[1] - 9])


class FR:
    def __init__(self):
        pass

    def minimize(self, x, f, grad, eps=0.01, max_iter=500000):
        k = 0
        x1 = x  # x[k-1]
        x2 = x  # x[k]
        pk = np.array([])
        for i in range(max_iter):
            g = grad(x2)
            if self.get_module(g) > eps:
                if k == 0:
                    p = -grad(x2)
                else:
                    b = self.get_beta(x1, x2)
                    p = -grad(x2) + np.dot(b, pk)
            else:
                return x1
            h = self.get_step(f, grad, p, x2)
            x1 = x2
            x2 = x1 - h * p
            pk = p
            k += 1
        return x2

    def get_beta(self, x1, x2):
        tmp1 = 0.0
        tmp2 = 0.0
        for i in range(len(x1)):
            tmp1 += x1[i] * x1[i]
            tmp2 += x2[i] * x2[i]
        return tmp2 / tmp1

    def get_step(self, f, grad, p, x):
        hes = self.get_hessian(f, grad, x)
        g = grad(x)
        tmp = np.dot(np.dot(hes, p), p)
        tmph = tmp
        tmp = np.dot(g, p)
        return tmp / tmph

    def get_hessian(self, f, grad, x):
        # исправить
        return np.array([[2, 1], [1, 2]])

    def get_module(self, x):
        tmp = 0.0
        for el in x:
            tmp += el * el
        return np.sqrt(tmp)


if __name__ == "__main__":
    opt = FR()
    x = opt.minimize(np.array([0, 1]), tf, dtf)
    print(x)
    print(tf(x))