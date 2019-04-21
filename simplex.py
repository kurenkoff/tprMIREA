import numpy as np
from math import fabs


class Simplex:
    # n - размерность задачи минимизации
    # m - длина ребра симплекса
    # eps - точность поиска
    def __init__(self, n, m, eps):
        self.m = m
        self.n = n
        self.eps = eps
        self.beta1 = m * (np.sqrt(n + 1) - 1) / (n * np.sqrt(n))
        self.beta2 = m * (np.sqrt(n + 1) + n - 1) / (n * np.sqrt(n))

    def minimize(self, fnc, zero_x, max_iterations=10):
        table = []
        zero_x.append(fnc(zero_x))
        table.append(zero_x)
        # инициализация таблицы
        for i in range(self.n):
            tmp_x = []
            for j in range(self.n):
                tmp_x.append(zero_x[j] + self.beta1 if i == j else zero_x[j] + self.beta2)
            tmp_x.append(fnc(tmp_x))
            table.append(tmp_x)
        print(table)
        xc = []  # костыль
        i = 1
        # Итерационный процесс
        while self.check_end(xc, table) and i <= max_iterations:
            im = self.max(table)
            new_x = self.iter(table, im, fnc)
            print("new_x = ", new_x)
            table.pop(im)
            table.append(new_x)
            xc = self.new_xc(table, fnc)
            print("table = ", table)
            i += 1
        # возврат минимального значения
        return self.min(table)

    # вовращает номер строки с максимальным значением оптимизируемой функции
    def max(self, table):
        im = 0
        for i in range(self.n):
            if table[i][self.n] > table[im][self.n]:
                im = i
        return im

    def min(self, table):
        im = 0
        for i in range(self.n):
            if table[i][self.n] < table[im][self.n]:
                im = i
        return table[im][self.n]

    # вычисление новой вершины симплекса
    def iter(self, table, im, fnc):
        xc = []
        for i in range(self.n):
            tmp_x = 0
            for j in range(len(table)):
                if j != im:
                    tmp_x += table[j][i]
            xc.append(tmp_x)
        xc.append(0)
        print("xc = ", str(xc))
        tmp_x = []
        for i in range(len(xc)):
            tmp_x.append(xc[i] - table[im][i])
        tmp_x.pop(self.n)
        tmp_x.append(fnc(tmp_x))
        return tmp_x

    # проверка окончания вычисления
    def check_end(self, xc, table):
        if not xc:
            return True
        for i in range(self.n + 1):
            if fabs(table[i][self.n] - xc[self.n]) > self.eps:
                return True
        return False

    # вовращает значение центра тяжести текущего симплекса
    def new_xc(self, table, fnc):
        xc = []
        for i in range(self.n):
            tmp_x = 0
            for j in range(self.n):
                    tmp_x += table[j][i]
            xc.append(tmp_x)
        xc = (np.array(xc) * (1 / 3)).tolist()
        xc.append(fnc(xc))
        return xc


def test_func(x):
    return 1.2 * x[0] * x[0] - 2.4 * x[1] - 1.6


def f(x):
    return x[0] ** 2 + x[0] * x[1] + x[1] ** 2 - 6 * x[0] - 9 * x[1]


s = Simplex(2, 0.25, 0.1)
print(s.minimize(f, [0, 0], max_iterations=1000))
print(f([1.08056640625, 3.95654296875]))