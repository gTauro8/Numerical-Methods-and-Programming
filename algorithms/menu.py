from tests_metods import *
import os
from welcome import welcome


def menu():
    while True:
        os.system("cls")
        print(welcome("Calcolo Numerico"))
        print("\nChoose service you want to use : ")
        print("""
        1 : Risoluzione di sistemi di equazioni Lineari Ab = x 
        2 : Interpolazione
        3 : Interpolazione -> calcolo dei nodi
        4 : Ricerca delle radici di funzione
        5 : Algoritmi per il calcolo integrale
        0 : Exit"""
              )
        choice = input("\nEnter your choice : ")

        if choice == '1':
            test_fattorizzazione(200)
        elif choice == '2':
            testInterpolazione(-2 * np.pi, 2 * np.pi, 60, 200, np.cos)
        elif choice == '3':
            test_nodi(-2 * np.pi, 2 * np.pi, 200, 60, np.cos)
        elif choice == '4':
            test_ricerca_radici()
        elif choice == '5':
            print("ciao2")
        elif choice == '0':
            exit()
        os.system("cls")
