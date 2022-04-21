import numpy as np


def epsilon_machine():
    eps = 1.0
    while eps + 1 > 1:
        eps /= 2
    eps *= 2

    return eps


def eliminazione_gauss(A, b):
    A = np.copy(A)
    b = np.copy(b)

    n = len(A)

    for j in range(n - 1):
        for i in range(j + 1, n):
            m = A[i, j] / A[j, j]

            for k in range(j + 1, n):
                A[i, k] = A[i, k] - m * A[j, k]
            b[i] = b[i] - m * b[j]

    return A, b


def sostituzione_indietro(U, b):
    n = len(U)
    x = np.zeros(n)

    if abs(np.prod(np.diag(U))) < epsilon_machine():
        print("Attenzione! Questa matrice potrebbe non avere soluzioni")
    else:
        for i in range(n - 1, -1, -1):
            S = 0

            for j in range(i + 1, n):
                S = S + U[i, j] * x[j]

            x[i] = (b[i] - S) / U[i, i]

    return x


def sostituzione_avanti(L, b):
    # Prendiamo il numero di righe
    n = L.shape[0]

    # Allochiamo spazio per il vettore soluzione
    y = np.zeros_like(b, dtype=np.double)

    # Applichiamo la sostituzione in avanti
    y[0] = b[0] / L[0, 0]

    # Cicliamo al contrario sulle righe(bottom up)
    # Iniziando dalla seconda all'ultima riga
    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    return y


def fattorizzazioneLU(A):
    n = len(A)

    U = A.copy()
    L = np.eye(n)

    for i in range(n):
        m = U[i + 1:, i] / U[i, i]
        L[i + 1:, i] = m
        U[i + 1:] = U[i + 1:] - m[:, np.newaxis] * U[i]

    return L, U


def fattorizzazione_doolittle(A):
    # fattorizzazione Lu usando la fattorizzazione Doolittle

    L = np.zeros_like(A)
    U = np.zeros_like(A)
    N = np.size(A, 0)

    for k in range(N):
        L[k, k] = 1
        U[k, k] = (A[k, k] - np.dot(L[k, :k], U[:k, k])) / L[k, k]
        for j in range(k + 1, N):
            U[k, j] = (A[k, j] - np.dot(L[k, :k], U[:k, j])) / L[k, k]
        for i in range(k + 1, N):
            L[i, k] = (A[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]

    return L, U


def matrice_crout(A):
    n = A.shape[0]

    U = np.zeros((n, n), dtype=np.double)
    L = np.zeros((n, n), dtype=np.double)

    for k in range(n):
        L[k, k] = A[k, k] - L[k, :] @ U[:, k]

        U[k, k:] = (A[k, k:] - L[k, :k] @ U[:k, k:]) / L[k, k]
        L[(k + 1):, k] = (A[(k + 1):, k] - L[(k + 1):, :] @ U[:, k]) / U[k, k]

    return L, U


def fattorizzazioneLU_pivoting(A):
    A = np.copy(A)
    n = len(A)

    indice = np.array(range(n))

    for j in range(n - 1):
        max_A = abs(A[j, j])

        indice_pivot = j

        for i in range(j + 1, n):
            if abs(A[i, j] > max_A):
                indice_pivot = i

        # possibile scambio di righe
        if indice_pivot > j:
            for k in range(n):
                A[indice_pivot, k], A[j, k] = A[j, k], A[indice_pivot, k]
            indice[indice_pivot], indice[j] = indice[j], indice[indice_pivot]

        # eleminazione della colonna j-esima

        for i in range(j + 1, n):
            A[i, j] = A[i, j] / A[j, j]

            for k in range(j + 1, n):
                A[i, k] = A[i, k] - A[i, j] * A[j, k]

    L = np.tril(A, - 1) + np.eye(n, n)
    U = np.tril(A)

    return L, U


def fattorizzazioneLU_pivoting_ottimizzato(A):
    A = np.copy(A)
    n = len(A)

    indice = np.array(range(n))

    for j in range(n - 1):
        # individuazione elemento pivot
        indice_pivot = np.argmax(abs(A[j: n, j])) + j

        # eventuale scambio di righe
        if indice_pivot > j:
            A[[indice_pivot, j]] = A[[j, indice_pivot], :]
            indice[[indice_pivot, j]] = indice[[j, indice_pivot]]

        # eliminazione della colonna j-esima
        for i in range(j + 1, n):
            A[i, j] = A[i, j] / A[j, j]
            A[i, j + 1: n] = A[i, j + 1: n] - A[i, j] * A[j, j + 1: n]

    L = np.tril(A, - 1) + np.eye(n, n)
    U = np.tril(A)

    return L, U


