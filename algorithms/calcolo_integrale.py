import numpy as np
import matplotlib.pyplot as plt


def formula_trapezio(f, a, b):
    return ((b - a) / 2) * (f(a) + f(b))


def formula_Simpson(f, a, b):
    t = np.linspace(a, b, 3)
    return ((t[2] - t[0]) / 6) * (f(t[0]) + 4 * f(t[1]) + f(t[2]))


def formula_Boole(f, a, b):
    t = np.linspace(a, b, 5)
    return float(((t[4] - t[0]) / 90) * (7 * f(t[0]) + 32 * f(t[1]) + 12 * f(t[2]) + 32 * f(t[3]) + 7 * (t[4])))

# generalizzazione di composizione per i metodi (TRAPEZIO, SIMPSON, BOOLE)


def formula_Composta(f, formula, a, b, N):
    s = np.linspace(a, b, N + 1)
    S = []

    for i in range(N):
        S.append(float(formula(f, s[i], [s + 1])))

    return sum(S)


# funzione di primo grado
def f1(x):
    return 8 * x - 4


def F1(x):
    return 4 * x ** 2 - 4 * x


# funzione di terzo grado


def f3(x):
    return 4 * x ** 3 - 3 * x ** 2 + x - 7


def F3(x):
    return x ** 4 - x ** 3 + x ** 2 / 2 - 7 * x


# funzione di quinto grado


def f5(x):
    return 9 * x ** 5 + 3 * x ** 4 + 2 * x ** 3 - 5 * x ** 2 + 11 * x - 32


def F5(x):
    return (3 / 2) * x ** 6 + (3 / 5) * x ** 5 + x ** 4 / 2 - (5 / 3) * x ** 3 + (11 / 2) * x ** 2 - 32 * x


def I(F, a, b):
    return F(b) - F(a)


a = - 10
b = 10

