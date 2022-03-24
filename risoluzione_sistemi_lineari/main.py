import matplotlib.pyplot as plt
import metodi_soluzione as ms
import metodi_fattorizzazione as mf
import time
import numpy as np


def test(N):
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


test(200)
