import sys

matrix_list = []
def printMatrix(matrix_list):
    for matrix in matrix_list:
        for row in matrix:
            for element in range(len(matrix[row])):
                if matrix[row][element] < 0:
                    print("", end="")
                else:
                    print(" ", end="")
                print(matrix[row][element], end=" "),
            print("\n")
        print("\n")

def matrixMultiplication(matrixOne, matrixTwo):
    matrixResult = {}
    if len(matrixTwo['0']) != len(matrixOne):
        print("row1 != col2. Exit!")
        sys.exit
    else:
        matrixResult = {}
        for i in range(len(matrixOne)):
            matrix_row = []
            for j in range(0,len(matrixTwo['0'])):
                matrix_row.append(0)
            matrixResult[str(i)] = matrix_row

        for row in range(len(matrixOne)):
            for column in range(0, len(matrixTwo['0'])):
                result = 0
                for k in range(len(matrixOne['0'])):
                    result += (int(matrixOne[str(row)][k]) * int(matrixTwo[str(k)][column]))
                matrixResult[str(row)][column] = result

    return matrixResult


def loadMatrix(file):
    with open(file, 'r') as f:
        matrix_dict = {}
        num_rows, num_cols = 0, 0  # inicijalna vrijednost
        incorrect_matrix = False

        for line in f:
            if incorrect_matrix:
                continue
            line = line.strip()  # uklanja bijele prostore sa kraja reda
            if not line:  # ako je red prazan, to označava kraj trenutne matrice
                num_rows = 0
                num_cols = 0
                if matrix_dict != {}:
                    matrix_list.append(matrix_dict)  # dodaje trenutnu matricu u listu
                matrix_dict = {}  # resetira rječnik za sljedeću matricu
                continue

            else:
                if num_rows == 0 and num_cols == 0:
                    incorrect_matrix = False
                    pom = line.split(' ')
                    num_rows = int(pom[0])
                    num_cols = int(pom[1])
                    for i in range(num_rows):
                        matrix_row = []
                        for j in range(num_cols):
                            matrix_row.append(0)
                        matrix_dict[str(i)] = matrix_row

                else:
                    pom = line.split(' ')
                    if int(pom[0]) > num_rows or int(pom[1]) > num_cols:
                        incorrect_matrix = True
                        matrix_dict = {}
                        continue
                    matrix_dict[str(int(pom[0]))][int(pom[1])] = int(pom[2])


loadMatrix("matrice.txt")
matrixMultiplication(matrixOne=matrix_list[0], matrixTwo=matrix_list[1])
printMatrix(matrix_list)

def main():
    # Provjera da li su unesena dva argumenta
    if len(sys.argv) != 3:
        print("Molim unesite dvije putanje: prva za datoteku s matricama, druga za datoteku s rezultatom.")
        return

    putanja_matrice = sys.argv[1]
    putanja_rezultata = sys.argv[2]

    # Ovdje bi se učitala matrica
    loadMatrix(putanja_matrice)

    # Računanje rezultata i pisanje u datoteku
    rezultat = matrixMultiplication(matrixOne=matrix_list[0], matrixTwo=matrix_list[1])
    with open(putanja_rezultata, 'w') as f:
        # Pretpostavka je da je rezultat neka tekstualna reprezentacija matrice
        f.write(str(rezultat))

if __name__ == "__main__":
    main()
