from math import sqrt, fabs


def fun(x):
    return x[0] ** 2 - x[0] * x[1] + 3 * x[1] ** 2 - x[0]


def simplex_search(n, m, eps, x, func):
    # initial process
    beta1 = ((sqrt(n + 1) - 1) / (n * sqrt(2))) * m
    beta2 = ((sqrt(n + 1) + n - 1) / (n * sqrt(2))) * m
    vect_x = list()
    vect_x.append(x)
    f_val = list()
    f_val.append(fun(x))
    for i in range(n):
        tmp_list = list()
        for j in range(n):
            tmp_x = 0
            if i == j:
                tmp_x = x[j] + beta1
            else:
                tmp_x = x[j] + beta2
            tmp_list.append(tmp_x)
        vect_x.append(tmp_list)
        f_val.append(fun(tmp_list))
    # iteration process
    return iter(n, m, vect_x, f_val, eps)


def iter(n, m, vect_x, f_val, eps):
    print(1)
    mi = f_val.index(max(f_val))
    fc = fxc(n, vect_x, mi)
    # make new point
    for i in range(n):
        vect_x[mi][i] = fc[i] - vect_x[mi][i]
    fx = f_val[mi]
    f_val[mi] = fun(vect_x[mi])
    if f_val[mi] >= fx:
        # reduction
        r = f_val.index(min(f_val))
        for i in range(n + 1):
            if i != r:
                vect_x[i] = reduction(vect_x[i], vect_x[r])
    xc = list()
    for i in range(n):
        tmp = 0
        for j in range(n + 1):
            tmp = vect_x[j][i]
        xc.append(tmp)
    f = fun(xc)
    for i in range(n + 1):
        if fabs(f_val[i] - f) >= eps:
            return iter(n, m, vect_x, f_val, eps)
    return vect_x[f_val.index(min(f_val))]


def fxc(n, vect_x, mi):
    fc = list()
    for i in range(n):
        tmp_x = 0
        for j in range(n + 1):
            if j != mi:
                tmp_x += vect_x[j][i]
        fc.append(tmp_x)
    return fc


def reduction(xi, xr):
    result = list()
    for i in range(len(xi)):
        result.append(xr[i] + 0.5 * (xi[i] - xr[i]))
    return result


if __name__ == "__main__":
    print(simplex_search(2, 0.25, 0.1, [0, 0], fun))

