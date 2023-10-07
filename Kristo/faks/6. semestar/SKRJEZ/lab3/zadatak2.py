import sys
import pandas

def main():
    # Provjera da li su unesena dva argumenta
    if len(sys.argv) != 2:
        print("Molim unesite putanju do datoteke s hipotezama")
        return

    putanja = sys.argv[1]

    stupci = ['Q' + str(i) + '0' for i in range(1, 10)]
    df = pandas.DataFrame(columns=stupci)

    brRetka = 1
    with open(putanja, 'r') as f:
        for line in f:
            line = line.strip()
            lista = []
            redak = sorted(line.split(" "))
            for i in range(0, 9):
                index = int(len(redak) * i * 0.1)
                lista.append(redak[index + 1])
            df.loc[str(brRetka).zfill(3)] = lista
            brRetka += 1

    # Definirajte zaglavlje
    header = '#' + '#'.join(df.columns)

    # Ispis zaglavlja
    print("Hyp", header)

    # Ispis svakog retka
    for row in df.itertuples():
        print('#'.join(map(str, row)))

if __name__ == "__main__":
    main()