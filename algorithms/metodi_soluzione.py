import numpy as np
import time
import metodi_fattorizzazione as mf


def soluzione_gauss(A, b):
    A = np.copy(A)
    b = np.copy(b)

    U, b = mf.eliminazione_gauss(A, b)
    x = mf.sostituzione_indietro(U, b)

    return x


def soluzione_cramer(A, b):
    x = []
    n = len(A)
    detA = np.linalg.det(A)

    for i in range(n):
        Ai = np.copy(A)
        Ai[:, i] = b
        x.append(float(np.linalg.det(Ai) / np.linalg.det(A)))

    return np.copy(list(x))


def soluzione_cramer_ottimizzato(A, b):
    x = []
    n = len(A)
    detA = np.linalg.det(A)

    for i in range(n):
        Ai = np.copy(A)
        Ai[:, i] = b
        x.append(float(np.linalg.det(Ai) / detA))

    return np.copy(list(x))


def soluzione_matrice_inversa(A, b):
    return np.dot(np.linalg.inv(A), b)


def soluzioneLU(A, b):
    L, U = mf.fattorizzazioneLU(A)

    y = mf.sostituzione_avanti(L, b)
    x = mf.sostituzione_indietro(U, y)

    return x


def soluzione_matrice_croute(A, b):
    L, U = mf.matrice_crout(A)

    y = mf.sostituzione_avanti(L, b)
    x = mf.sostituzione_indietro(U, y)
    return x


def soluzione_doolittle(A, b):
    L, U = mf.fattorizzazione_doolittle(A)

    y = mf.sostituzione_avanti(L, b)
    x = mf.sostituzione_indietro(U, y)
    return x


def soluzioneBuildIn(A, b):
    return np.linalg.solve(A, b)
