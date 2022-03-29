import math
import numpy as np


def f(x):
    return x ** 2 - 11


def df(x):
    return 2 + x


def metodo_Bisezioni_Successive(a: float, b: float, tolleranza: float, x_reale: float, fun):
    # Verifica presenza di radici
    fa = fun(a)
    fb = fun(b)

    c = None

    if fa * fb > 0:
        print('Errore: non garantita radice in [%f; %f]' % (a, b))
    else:
        n = math.ceil(math.log2((b - a) / tolleranza)) - 1
        e = np.zeros(n + 1)  # def dell'errore

        for k in range(n + 1):
            c = (a + b) / 2
            fc = fun(c)

            e[k] = abs(x_reale - c)
            if fa * fc < 0:
                b = c
            else:
                a = c
                fa = fc

    return c, e


def metodo_Newton(x0: float, tolleranza: float, kmax: int, x_reale: float, fun, dfun):
    fx0 = fun(x0)
    dfx0 = dfun(x0)
    err = []

    iterazioni = 0
    stop = 0

    while not stop and iterazioni < kmax:
        x1 = x0 - fx0 / dfx0
        fx1 = fun(x1)

        stop = abs(fx1) + abs(x1 - x0) / abs(x1) < tolleranza / 5

        err.append(abs(x_reale - x1))

        iterazioni += 1

        if not stop:
            x0 = x1
            fx0 = fx1
            dfx0 = dfun(x0)

    if not stop:
        print("Accuratezza del metodo raggiunta in %d iterazioni" % (int(iterazioni)))

    return x1, err


def metodo_Secanti(x0, x1, tolleranza, k_max, x_reale: float, fun):
    fx0 = fun(x0)
    fx1 = fun(x1)

    err = []

    iterazioni = 0
    stop = 0

    while not stop and iterazioni < k_max:
        x2 = x1 - ((fx1*(x1 - x0)) / (fx1 - fx0))
        fx2 = fun(x2)

        stop = abs(fx2) + abs(x2 - x1) / abs(x2) < tolleranza / 5

        err.append(abs(x_reale - x2))

        iterazioni += 1

        if not stop:
            x0 = x1
            fx0 = fx1
            x1 = x2
            fx1 = fx2

    if not stop:
        print("Accuratezza del metodo raggiunta in %d iterazioni" %(int(iterazioni)))

    return x1, err


def metodo_Corde(x0: float, m, tolleranza, kmax, x_reale: float, fun):
    fx0 = fun(x0)
    err = []

    iterazioni = 0
    stop = 0

    while not stop and iterazioni < kmax:
        x1 = x0 - fx0/m
        fx1 = fun(x1)

        stop = abs(fx1) + abs(x1 - x0) / abs(x1) < tolleranza / 5
        err.append(abs(x_reale - 1))

        iterazioni += 1

        if not stop:
            print("Accuratezza del metodo raggiunta in %d iterazioni" %(int(iterazioni)))

        return x1, err


