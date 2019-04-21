import numpy as np


class PatternSearch:
    def __init__(self):
        pass

    # minimize возвращает точку в котором функция принимает наименьшее значение
    # f - целевая функция
    # x - np.array. начальная точка
    # h - шаг алгоритма
    # d - коэффициент уменьшения шага
    # m - ускоряющий множитель
    # eps - точность поиска
    def minimize(self, f, x, h, d, m, eps=0.01, max_iterations=100):
        x1 = x.astype("float")
        fx1 = f(x)
        for i in range(max_iterations):
            x2, h1 = self.search_elementwise(f, x1, h, d)
            h = h1
            x1 = self.pattern_search(f, x2, x1, m)
            print("x1 = ", x1, " x2 = ", x2)
            if h < eps:
                return x1

    # search реализует иследующий поиск. Шаг прибавляется к вектору
    def search(self, f, x1, h, d):
        x2 = x1 + h

        if f(x2) < f(x1):
            if np.array_equal(x1, x2):
                h = h / d
                return self.search(f, x1, h, d)
            return x2, h
        else:
            x2 = x2 - 2 * h
            if f(x2) < f(x1):
                if np.array_equal(x1, x2):
                    h = h / d
                    return self.search(f, x1, h, d)
                return x2, h
            else:
                x2 = x2 + h
                if np.array_equal(x1, x2):
                    h = h / d
                    return self.search(f, x1, h, d)
                else:
                    return x2, h

    # search_elementwise иследующий поиск. Шаг прибавлсяется к каждому компоненту вектора
    def search_elementwise(self, f, x1, h, d):
        x2 = np.array(x1)
        for i in range(len(x1)):
            x2[i] += h
            if f(x2) >= f(x1):
                x2[i] = x2[i] - 2 * h
                if f(x2) >= f(x1):
                    x2[i] = x2[i] + h
        if np.array_equal(x1, x2):
            h = h / d
            return self.search_elementwise(f, x1, h, d)
        return x2, h

    # pattern_search поиск по образцу
    def pattern_search(self, f, x1, x0, m):
        xp = x1 + m * (x1 - x0)
        try:
            if f(xp) < f(x1):
                return xp
        except IndexError:
            pass
        return x1


def test_f1(x):
    return x[0] ** 2 - x[0] * x[1] + 3 * x[1] ** 2 - x[0]


def tf(x):
    return x[0] ** 2 + x[0] * x[1] + x[1] ** 2 - 6 * x[0] - 9 * x[1]


if __name__ == "__main__":
    ps = PatternSearch()
    f = tf
    x = np.array([0., 0.])
    print(x)
    h = 0.2
    d = 2
    eps = 0.1
    m = 0.5
    print(f(ps.minimize(f, x, h, d, m, eps)))

