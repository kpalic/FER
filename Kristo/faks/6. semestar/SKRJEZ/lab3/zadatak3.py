import os
import sys

def main():
    if len(sys.argv) != 2:
        print("Molim unesite putanju kao argument.")
        return

    putanja = sys.argv[1]
    studenti = {}
    vjezbe = set()

    # Učitaj podatke o studentima
    with open(os.path.join(putanja, 'studenti.txt'), 'r') as f:
        brRetka = 0
        for line in f:
            if brRetka == 0:
                brRetka += 1
                continue
            mat_broj, prezime, ime = line.strip().split(' ', 2)
            studenti[mat_broj] = {'ime': f"{prezime}, {ime}"}

    # Učitaj podatke o laboratorijskim vježbama
    for datoteka in os.listdir(putanja):
        if datoteka.startswith('Lab_'):
            broj_vjezbe = datoteka.split('_')[1].split('.')[0]
            vjezbe.add(broj_vjezbe)
            with open(os.path.join(putanja, datoteka), 'r') as f:
                for line in f:
                    mat_broj, bodovi = line.strip().split(' ', 1)
                    if mat_broj in studenti:
                        if broj_vjezbe in studenti[mat_broj]:
                            print(f"Upozorenje: student {mat_broj} je već evidentiran na vježbi {broj_vjezbe}.")
                        else:
                            studenti[mat_broj][broj_vjezbe] = bodovi

    # Ispisuj podatke
    print("JMBAG Prezime, Ime", ' '.join(f"L{vjezba}" for vjezba in sorted(vjezbe)))
    for mat_broj, podaci in studenti.items():
        redak = [f"{mat_broj} {podaci['ime']}"]
        for vjezba in sorted(vjezbe):
            redak.append(podaci.get(vjezba, '-'))
        print(' '.join(redak))

if __name__ == "__main__":
    main()
