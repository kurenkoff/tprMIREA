import numpy as np


def tf(x):
    return x[0] ** 2 + x[0] * x[1] + x[1] ** 2 - 6 * x[0] - 9 * x[1]


def test_func(x):
    return 1.2 * x[0] * x[0] - 2.4 * x[1] - 1.6


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


class NelderMead():
    def __init__(self):
        pass

    def minimize(self, f, alpha=1, beta=0.5, gamma=2, max_iterations=100):

        v1 = np.array([1., 0])
        v2 = np.array([0, 0])
        v3 = np.array([0, 1.])

        print("f(v1) = ", f(v1))
        print("f(v2) = ", f(v2))
        print("f(v3) = ", f(v3))

        print("==============================================================")

        for i in range(max_iterations):
            print("Итерация №", i)
            sdict = {totuple(v1): f(v1), totuple(v2): f(v2), totuple(v3): f(v3)}
            points = sorted(sdict.items(), key=lambda item: item[1])

            best = np.array(points[0][0])
            good = np.array(points[1][0])
            worst = np.array(points[2][0])
            print(f(best), f(good), f(worst))
            # отражение
            mid = (best + good) / 2
            print(mid)
            xr = mid + alpha * (mid - worst)
            print(xr)
            if f(xr) < f(good):
                worst = xr
            else:
                if f(xr) < f(worst):
                    worst = xr
                c = (worst + mid) / 2
                if f(c) < f(worst):
                    worst = c
            if f(xr) < f(best):
                # растяжение
                xe = mid + gamma * (xr - mid)
                if f(xe) < f(xr):
                    worst = xe
                else:
                    worst = xr
            if f(xr) > f(good):
                #
                xc = mid + beta * (worst - mid)
                if f(xc) < f(worst):
                    worst = xc
            v1 = worst
            v2 = good
            v3 = best
            print(v1, v2, v3)
            print("==============================================================")
        return best


if __name__ == "__main__":
    n = NelderMead()
    point = n.minimize(tf, max_iterations=100)
    print("РЕЗУЛЬТАТ\n", point)
    print(tf(point))
