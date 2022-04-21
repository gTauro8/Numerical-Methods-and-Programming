import numpy as np
from metodi_soluzione import soluzione_gauss
from metodi_fattorizzazione import epsilon_machine


def coefficienti_indeterminati(xn: np.ndarray, yn: np.ndarray, x):
    n = len(xn)
    p = np.zeros(len(x))

    # costruzione della matrice di Vandermonden
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            A[i, j] = np.power(xn[i], j)

    # calcoliamo i coefficienti indeterminati
    c = soluzione_gauss(A, yn)

    for i in range(len(x)):
        for n in range(len(c)):
            p[i] += c[n] * (np.power(x[i], n))

    return p


def z_coefficiente(xn: np.ndarray, yn: np.ndarray) -> np.ndarray:
    n: int = len(xn)
    X: np.ndarray = np.eye(n)

    for i in range(n):
        for j in range(n):
            if j > i:
                X[i, j] = xn[i] - xn[j]
            elif j < i:
                X[i, j] = - X[j, i]
    zn: np.ndarray = np.zeros(n)
    for j in range(n):
        zn[j] = yn[j] / np.prod(X[j, :])

    return zn


def calcola_Lagrange(x: float, xn: np.ndarray, yn: np.ndarray, zn: np.ndarray) -> float:
    trova_nodi = abs(x - xn) < epsilon_machine()

    if True in trova_nodi:
        temp = np.flatnonzero(trova_nodi == True)
        j = temp[0]
        pn = yn[j]
    else:
        n = len(xn)
        S = 0
        for j in range(n):
            S = S + zn[j] / (x - xn[j])
        pn = np.prod(x - xn) * S

    return pn


def metodo_Lagrange(xn: np.ndarray, yn: np.ndarray, x: np.ndarray) -> np.ndarray:
    len_x: int = len(x)

    # Calcolo i coefficienti formula baricentrica
    zn: np.ndarray = z_coefficiente(xn, yn)

    # calcolo polinomio interpolazione nei punti di x
    p = np.zeros(len_x)
    for i in range(len_x):
        p[i] = calcola_Lagrange(x[i], xn, yn, zn)

    return p


def differenze_finite(xn: np.ndarray, yn: np.ndarray) -> np.ndarray:
    d: np.ndarray = np.copy(yn)
    n: int = len(yn)

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            d[i] = (d[i] - d[i - 1]) / (xn[i] - xn[i - j])

    return d


def calcola_Newton(x: float, xn: np.ndarray, d: np.ndarray) -> float:
    n: int = len(xn) - 1
    p: float = d[n]

    for i in range(n - 1, -1, -1):
        p = d[i] + p * (x - xn[i])

    return p


def metodo_Newton(xn: np.ndarray, yn: np.ndarray, x: np.ndarray) -> np.ndarray:
    len_x: int = len(x)
    d: np.ndarray = differenze_finite(xn, yn)
    p: np.ndarray = np.zeros(len_x)

    for i in range(len_x):
        p[i] = calcola_Newton(x[i], xn, d)

    return p


def nodi_Chebyshev(a: float, b: float, n: int) -> np.ndarray:
    x = np.empty(n)
    for i in range(n):
        x[i] = a + (np.cos((2*i+1)/(2*n+2)*np.pi) + 1) * (b-a)/2
    return x
