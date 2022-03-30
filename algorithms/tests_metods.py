import math
import matplotlib.pyplot as plt
import metodi_soluzione as ms
import metodi_fattorizzazione as mf
import funzioni_interpolazione as fi
import ricerca_raduci_funzione as rrf
import time
import numpy as np
from welcome import welcome
import calcolo_integrale as ci


def test_fattorizzazione(N):
    print(welcome("Ab = x risoluzioni"))
    t_Soluzione_Gauss = []
    t_soluzione_ALU_senza_Pivot = []
    t_soluzione_Cramer = []
    t_soluzione_Cramer_ottimizzato = []

    t_soluzione_matrice_inversa = []
    t_soluzione_doolittle = []
    t_soluzione_crout = []
    t_build_in = []

    for n in range(2, N + 1, 5):
        A = (2 * np.random.random((n, n)) - 1) * 10
        b = (2 * np.random.random(n) - 1) * 10

        # effettuo una verifica sul determinante osservando che
        # sia !=0 e quindi la matrice non sia singolare

        while np.linalg.det(A) < mf.epsilon_machine():
            A = (2 * np.random.random((n, n)) - 1) * 10

            print("Matrice rigenerata")

        # eliminazione di gauss
        tempo_iniziale = time.time()
        x_gauss = ms.soluzione_gauss(A, b)
        tempo_finale = time.time()

        t_Soluzione_Gauss.append(float(tempo_finale - tempo_iniziale))

        print("Durata esecuzione con matrice %dx%d: %f (soluzione con MEG)"
              % (n, n, t_Soluzione_Gauss[len(t_Soluzione_Gauss) - 1]))

        # cramer
        tempo_iniziale = time.time()
        x_cramer = ms.soluzione_cramer(A, b)
        tempo_finale = time.time()

        t_soluzione_Cramer.append(float(tempo_finale - tempo_iniziale))

        print("Durata esecuzione con matrice %dx%d: %f (soluzione con METODO CRAMER)"
              % (n, n, t_soluzione_Cramer[len(t_soluzione_Cramer) - 1]))

        # cramer ottimizzato
        tempo_iniziale = time.time()
        x_cramer_ott = ms.soluzione_cramer_ottimizzato(A, b)
        tempo_finale = time.time()

        t_soluzione_Cramer_ottimizzato.append(float(tempo_finale - tempo_iniziale))

        print("Durata esecuzione con matrice %dx%d: %f (soluzione con METODO CRAMER OTTIMIZZATO)"
              % (n, n, t_soluzione_Cramer_ottimizzato[len(t_soluzione_Cramer_ottimizzato) - 1]))

        # matrice inversa
        tempo_iniziale = time.time()
        x_matrice_inversa = ms.soluzione_matrice_inversa(A, b)
        tempo_finale = time.time()

        t_soluzione_matrice_inversa.append(float(tempo_finale - tempo_iniziale))

        print("Durata esecuzione con matrice %dx%d: %f (soluzione con METODO MATRICE INVERSA)"
              % (n, n, t_soluzione_matrice_inversa[len(t_soluzione_Cramer_ottimizzato) - 1]))

        # alu senza pivot
        tempo_iniziale = time.time()
        x_soluzione_ALU_senza_Pivot = ms.soluzioneLU(A, b)
        tempo_finale = time.time()

        t_soluzione_ALU_senza_Pivot.append(float(tempo_finale - tempo_iniziale))

        print("Durata esecuzione con matrice %dx%d: %f (soluzione con METODO FATTORIZZAZIONE ALU SENZA PIVOT)"
              % (n, n, t_soluzione_ALU_senza_Pivot[len(t_soluzione_ALU_senza_Pivot) - 1]))

        # metodo numpy solve
        tempo_iniziale = time.time()
        x_soluzione_Build_In = ms.soluzioneBuildIn(A, b)
        tempo_finale = time.time()

        t_build_in.append(float(tempo_finale - tempo_iniziale))

        print("Durata esecuzione con matrice %dx%d: %f (soluzione con no.linalg.solve)"
              % (n, n, t_build_in[len(t_build_in) - 1]))

        # metodo matrice crout
        tempo_iniziale = time.time()
        x_soluzione_crout = ms.soluzione_matrice_croute(A, b)
        tempo_finale = time.time()

        t_soluzione_crout.append(float(tempo_finale - tempo_iniziale))

        print("Durata esecuzione con matrice %dx%d: %f (soluzione con matrici Crout)"
              % (n, n, t_soluzione_crout[len(t_soluzione_crout) - 1]))

        # metodo matrice crout
        tempo_iniziale = time.time()
        x_soluzione_doolittle = ms.soluzione_doolittle(A, b)
        tempo_finale = time.time()

        t_soluzione_doolittle.append(float(tempo_finale - tempo_iniziale))

        print("Durata esecuzione con matrice %dx%d: %f (soluzione con METODO DOOLITTLE)"
              % (n, n, t_soluzione_doolittle[len(t_soluzione_doolittle) - 1]))

    plt.figure(1)

    plt.plot(range(2, N + 1, 5), t_Soluzione_Gauss, 'k-',
             label='SOLUZIONE CON MEG')

    plt.plot(range(2, N + 1, 5), t_soluzione_ALU_senza_Pivot, 'y-',
             label='SOLUZIONE CON METODO A=LU')

    plt.xlabel('N')
    plt.ylabel('Tempo')
    plt.legend()
    plt.show()

    plt.plot(range(2, N + 1, 5), t_soluzione_Cramer, 'r-',
             label='SOLUZIONE CON CRAMER')

    plt.plot(range(2, N + 1, 5), t_soluzione_Cramer_ottimizzato, 'g-',
             label='SOLUZIONE CON CRAMER OTTIMIZZATO')

    plt.xlabel('N')
    plt.ylabel('Tempo')
    plt.legend()
    plt.show()

    plt.plot(range(2, N + 1, 5), t_soluzione_matrice_inversa, 'b-',
             label='SOLUZIONE CON MATRICE INVERSA')

    plt.plot(range(2, N + 1, 5), t_build_in, 'brown',
             label='SOLUZIONE CON METODO np.linalg.solve')

    plt.xlabel('N')
    plt.ylabel('Tempo')
    plt.legend()
    plt.show()

    plt.plot(range(2, N + 1, 5), t_soluzione_ALU_senza_Pivot, 'red',
             label='SOLUZIONE CON MATRICE A=LU')

    plt.plot(range(2, N + 1, 5), t_build_in, 'brown',
             label='SOLUZIONE CON METODO np.linalg.solve')

    plt.xlabel('N')
    plt.ylabel('Tempo')
    plt.legend()
    plt.show()

    plt.plot(range(2, N + 1, 5), t_soluzione_ALU_senza_Pivot, 'red',
             label='SOLUZIONE CON MATRICE A=LU')

    plt.plot(range(2, N + 1, 5), t_soluzione_crout, 'blue',
             label='SOLUZIONE CON MATRICE CROUT')

    plt.plot(range(2, N + 1, 5), t_soluzione_doolittle, 'green',
             label='SOLUZIONE CON METODO DOOLITTLE')

    plt.plot(range(2, N + 1, 5), t_build_in, 'brown',
             label='SOLUZIONE CON METODO np.linalg.solve')

    plt.xlabel('N')
    plt.ylabel('Tempo')
    plt.legend()
    plt.show()


###################################################################
# PARAMETRI IN INPUT
###################################################################
# a: float = Punto sx intervallo di interpolazione
# b: float = Punto dx intervallo di interpolazione
#
# n: int = Grado del polinomio
# nx: int = Nro di x da calcolare
#
# fun: function = Riferimento a funzione da interpolare
#
###################################################################


def testInterpolazione(a: float, b: float, n: int, nx: int, fun: type('function')):
    print(welcome("Algoritmi di interpolazione"))
    # Grado ed i nodi di interpolazione
    xn: np.ndarray = np.linspace(a, b, n + 1)
    yn: np.ndarray = fun(xn)

    # Creazione dell'insieme delle x su cui testare il polinomio di interpolazione
    x: np.ndarray = np.linspace(a, b, nx)

    # Calcolo della funzione per ogni xi in x
    fx: np.ndarray = fun(x)

    # Calcolo delle x con i polinomi di interpolazione (Lagrange, Newton)
    px_lagrange: np.ndarray = fi.metodo_Lagrange(xn, yn, x)
    px_newton: np.ndarray = fi.metodo_Newton(xn, yn, x)

    plt.plot(1)
    plt.plot(x, px_lagrange, 'tab:orange', label='Lagrange')
    plt.plot(x, px_newton, 'tab:blu', label='Newton')
    plt.plot(x, fx, 'k--', label='f(x) = cos(x)')
    plt.plot(xn, yn, 'ro')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    plt.plot(2)
    plt.plot(x, abs(fx - px_lagrange), "blue", label='|f(x) - Pn_Lagrange(x)|')
    plt.plot(x, abs(fx - px_newton), "red", label='|f(x) - Pn_Newton(x)|')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def test_nodi(a: float, b: float, nx: int, nmax: int, fun: type('function')):
    print(welcome("Calcolo dei nodi"))
    x = np.linspace(a, b, nx)
    fx = fun(x)

    resto_eq_lagrange = np.zeros(nmax)
    resto_ch_lagrange = np.zeros(nmax)

    resto_eq_newton = np.zeros(nmax)
    resto_ch_newton = np.zeros(nmax)

    for n in range(nmax):
        xn_eq = np.linspace(a, b, n + 1)
        yn_eq = fun(xn_eq)

        px_eq_lagrange = fi.metodo_Lagrange(xn_eq, yn_eq, x)
        px_eq_newton = fi.metodo_Newton(xn_eq, yn_eq, x)

        xn_ch = fi.nodi_Chebyshev(a, b, n + 1)
        yn_ch = fun(xn_ch)

        px_ch_lagrange = fi.metodo_Lagrange(xn_ch, yn_ch, x)
        px_ch_newton = fi.metodo_Newton(xn_ch, yn_ch, x)

        resto_eq_lagrange[n] = max(abs(fx - px_eq_lagrange))
        resto_ch_lagrange[n] = max(abs(fx - px_ch_lagrange))

        resto_eq_newton[n] = max(abs(fx - px_eq_newton))
        resto_ch_newton[n] = max(abs(fx - px_ch_newton))

    plt.figure(3)
    plt.semilogy(range(nmax), resto_eq_lagrange, "red", label="MAX RESTO NODI EQUILIBRATI (Lagrange)")
    plt.semilogy(range(nmax), resto_ch_lagrange, "green", label="MAX RESTO NODI DI CHEBISHEV (Lagrange)")

    plt.semilogy(range(nmax), resto_eq_lagrange, "blue", label="MAX RESTO NODI EQUILIBRATI (Newton)")
    plt.semilogy(range(nmax), resto_ch_lagrange, "black", label="MAX RESTO NODI DI CHEBISHEV (Newton)")

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def vettore_standard(err, l_max):
    v = np.array([i - i for i in range(l_max)], dtype=float)

    for i in range(len(err)):
        v[i] = err[i]
    return np.array(v)


def test_ricerca_radici():
    # soluzione reale del problema

    x_reale = math.sqrt(11)

    # numero massimo iterazioni
    k_max = 35

    # tolleranza per l'arresto del criterio

    tolleranza = mf.epsilon_machine()

    # estremi dell'intervallo

    a = -2
    b = 4

    # ora calcolo la soluzione e il vettore degli errori per il metodo delle Bisezioni Successive
    sol_BS, err_BS = rrf.metodo_Bisezioni_Successive(a, b, tolleranza, x_reale, rrf.f)

    # punto iniziale del metodo di newton
    x0 = 3

    sol_New, err_New = rrf.metodo_Newton(x0, tolleranza, k_max, x_reale, rrf.f, rrf.df)

    # punti iniziali del metodo delle secanti

    x0 = 1
    x1 = 3

    sol_Sec, err_Sec = rrf.metodo_Secanti(x0, x1, tolleranza, k_max, x_reale, rrf.f)

    # punti iniziali del metodo Corde
    x0 = 0
    m = 4

    sol_Corde, err_Corde = rrf.metodo_Corde(x0, m, tolleranza, k_max, x_reale, rrf.f)

    len_max_err = max(len(err_BS), len(err_New), len(err_Sec), len(err_Corde))

    err_BS = vettore_standard(err_BS, len_max_err)
    err_New = vettore_standard(err_New, len_max_err)
    err_Sec = vettore_standard(err_Sec, len_max_err)
    err_Corde = vettore_standard(err_Corde, len_max_err)

    plt.plot(1)
    plt.semilogy(range(len_max_err), err_BS, "orange", label="Metodo delle Bisezioni Successive")
    plt.semilogy(range(len_max_err), err_New, "green", label="Metodo di Newton")
    plt.semilogy(range(len_max_err), err_Sec, "red", label="Metodo delle Secanti")
    plt.semilogy(range(len_max_err), err_Corde, "blue", label="Metodo delle Corde")

    plt.legend()
    plt.xlabel("Nro. Iterazioni")
    plt.ylabel('Errore')
    plt.show()


def test_calcolo_integrale(f, F, a, b, N_INTERVALLI: int , grado: int):
    err_Trapezio = []
    err_Simpson = []
    err_Boole = []

    I_REALE = ci.I(F, a, b)

    for N in range(N_INTERVALLI):
        IC_TRAPEZIO = ci.formula_Composta(f, ci.formula_trapezio, a, b, N)
        IC_SIMPSON = ci.formula_Composta(f, ci.formula_Simpson, a, b, N)
        IC_BOOLE = ci.formula_Composta(f, ci.formula_Boole, a, b, N)

        err_T = abs(I_REALE - IC_TRAPEZIO)
        err_S = abs(I_REALE - IC_SIMPSON)
        err_B = abs(I_REALE - IC_BOOLE)

        err_Trapezio.append(err_T)
        err_Simpson.append(err_S)
        err_Boole.append(err_B)

        print("Intervallo [a, b] diviso in " + str(N) + "in sottointervalli")
        print("Errore formula COMPOSTA TRAPEZIO: |I - TC| = %f" % err_T)
        print("Errore formula COMPOSTA SIMPSON: |I - SC| = %f" % err_S)
        print("Errore formula COMPOSTA BOOLE: |I - BC| = %f" % err_B)

        plt.figure(1)
        plt.semilogy(range(N_INTERVALLI), err_Trapezio, 'b-*', label='Trapezio composto')
        plt.semilogy(range(N_INTERVALLI), err_Simpson, 'r-*', label='Simpson composto')
        plt.semilogy(range(N_INTERVALLI), err_Boole, 'g-*', label='Boole composto')

        plt.xlabel('N')
        plt.ylabel('Errore')
        plt.legend()
        plt.title('Errore al variare di N (f(x) di' + str(grado) + "grado)")


def test_funzioni_calcolo_integrale():
    test_calcolo_integrale(ci.f1, ci.F1, ci.a, ci.b, 200, 1)
    print("----------------------------------------------------------------\n")
    test_calcolo_integrale(ci.f3, ci.F3, ci.a, ci.b, 200, 3)
    print("----------------------------------------------------------------\n")
    test_calcolo_integrale(ci.f5, ci.F5, ci.a, ci.b, 200, 5)
    print("----------------------------------------------------------------\n")
