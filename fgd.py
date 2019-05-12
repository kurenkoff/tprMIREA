import numpy as np


def tf(x):
    return x[0] ** 2 + x[0] * x[1] + x[1] ** 2 - 6 * x[0] - 9 * x[1]


def dtf(x):
    return np.array([2 * x[0] + x[1] - 6, x[0] + 2 * x[1] - 9])


class FGDescent:
    def __int__(self):
        pass

    def minimize(self, x, f, grad, eps=0.01):
        # одна итерация
        while True:
            g = grad(x)
            if self.get_module(x) > eps:
                h = self.get_step(f, grad, x)
                x = x - h * grad(x)
            else:
                return x

    def get_step(self, f, grad, x):
        hes = self.get_hessian(f, grad, x)
        g = grad(x)
        gt = grad(x).T
        tmp = g * hes * gt
        tmph = tmp[0][0]
        tmp = g * g.T
        return tmp / tmph

    def get_hessian(self, f, grad, x):
        return np.array([[2, 1], [1, 2]])

    def get_module(self, x):
        tmp = 0.0
        for el in x:
            tmp += el * el
        return np.sqrt(tmp)


if __name__ == "__main__":
    a = FGDescent()
    print(a.minimize(np.array([1, 1]), tf, dtf))
    


