import numpy as np


def tf(x):
    return x[0] ** 2 + x[0] * x[1] + x[1] ** 2 - 6 * x[0] - 9 * x[1]

def dtf(x):
    return np.array([2 * x[0] + x[1] - 6, x[0] + 2 * x[1] - 9])



class Newton():
    def __init__(self):
        pass
    
    def minimize(self, x, f,  df, eps=0.1, max_iter=1000):
        for i in range(max_iter):
            grad = df(x)
            xk = np.array([])
            if self.get_module(x) > eps:
                H = self.get_hessian(f, grad, x)
                if self.is_pos_def(H):
                    xk = x - np.dot(np.linalg.inv(H), grad)
                else:
                    hk = self.get_step(f, grad, x)
                    xk = x - hk * grad
            else:
                return x
            x = xk
            print("x[{0}] = {1}]".format(i + 1, x))
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
        print(x)
        tmp = 0.0
        for el in x:
            tmp += el * el
        return np.sqrt(tmp)

    def is_pos_def(self, A):
        if np.allclose(A, A.T):
            try:
                np.linalg.cholesky(A)
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            return False



print(1)
n = Newton()
x = np.array([1, 1])
r_x = n.minimize(x, tf, dtf)
print(r_x)
print(tf(r_x))