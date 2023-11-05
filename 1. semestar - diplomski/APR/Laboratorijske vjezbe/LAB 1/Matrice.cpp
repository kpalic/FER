#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <filesystem>

#define EPSILON 10e-9

using namespace std;

class Matrix 
{
    private:
        vector<vector<double>> values;
        int rowNum;
        int colNum;
    public:
        Matrix(int rowNum, int colNum) {
            for (int i = 0; i < rowNum; i++) {
                vector<double> row(colNum, 0);
                values.push_back(row);
            }
            this->rowNum = rowNum;
            this->colNum = colNum;
        }
        Matrix(int rowNum, int colNum, vector<vector<double>> values) : 
            rowNum(rowNum), colNum(colNum), values(values) {};
        Matrix(const Matrix* matrix) {
            rowNum = matrix->rowNum;
            colNum = matrix->colNum;
            values = matrix->values;
        }
        Matrix(string file) {
            fstream matrixFile(file);
            string line;
            if (matrixFile.is_open()) {
                vector<vector<double>> newValues;
                while (getline(matrixFile, line)) {
                    stringstream ss(line);
                    vector<double> row;
                    double value;
                    while (ss >> value) {
                        row.push_back(value);
                    }
                    newValues.push_back(row);
                }
                values = newValues;
                rowNum = newValues.size();
                colNum = newValues[0].size();
            }
            else {
                cout << "Error: file not found" << endl;
            }
        }
        
        // OPERATORS
        Matrix& operator= (const Matrix& matrix) {
            if (this == &matrix) {
                return *this;
            }
            else {
                rowNum = matrix.rowNum;
                colNum = matrix.colNum;
                values = matrix.values;
                return *this;
            }
        }
        bool operator==(const Matrix& matrix) {
            if (rowNum == matrix.rowNum &&
                colNum == matrix.colNum == 0) {
                    return false;
            }
            for (int i = 0; i < rowNum; i++) {
                for (int j = 0; j < colNum; j++) {
                    if (abs(this->values[i][j] - matrix.values[i][j]) > EPSILON) {
                        return false;
                    }
                } 
            }
            return true;
        }

        Matrix operator+ (const Matrix& matrix) {
            if (rowNum != matrix.rowNum || colNum != matrix.colNum) {
                cout << "Error: matrix sizes do not match" << endl;
                return nullptr;
            }
            else {
                Matrix newMatrix(rowNum, colNum);
                for (int i = 0; i < rowNum; i++) {
                    for (int j = 0; j < colNum; j++) {
                        newMatrix.values[i][j] = this->values[i][j] + matrix.values[i][j];
                    }
                }
                return newMatrix;
            }
        }
        Matrix operator+= (const Matrix& matrix) {
            if (rowNum != matrix.rowNum || colNum != matrix.colNum) {
                cout << "Error: matrix sizes do not match" << endl;
                return nullptr;
            }
            else {
                for (int i = 0; i < rowNum; i++) {
                    for (int j = 0; j < colNum; j++) {
                        values[i][j] += matrix.values[i][j];
                    }
                }
                return *this;
            }
        }

        Matrix operator- (const Matrix& matrix) {
            if (rowNum != matrix.rowNum || colNum != matrix.colNum) {
                cout << "Error: matrix sizes do not match" << endl;
                return nullptr;
            }
            else {
                Matrix newMatrix(rowNum, colNum);
                for (int i = 0; i < rowNum; i++) {
                    for (int j = 0; j < colNum; j++) {
                        newMatrix.values[i][j] = this->values[i][j] - matrix.values[i][j];
                    }
                }
                return newMatrix;
            }
        }
        Matrix operator-= (const Matrix& matrix) {
            if (rowNum != matrix.rowNum || colNum != matrix.colNum) {
                cout << "Error: matrix sizes do not match" << endl;
                return nullptr;
            }
            else {
                for (int i = 0; i < rowNum; i++) {
                    for (int j = 0; j < colNum; j++) {
                        values[i][j] -= matrix.values[i][j];
                    }
                }
                return *this;
            }
        }
        
        Matrix operator* (const Matrix& matrix) {
            if (colNum != matrix.rowNum) {
                cout << "Error: colNums of 1st matrix != rowNums of 2nd matrix" << endl;
                return nullptr;
            }
            else {
                Matrix newMatrix(rowNum, matrix.colNum);
                vector<vector<double>> newValues;
                for (int i = 0; i < rowNum; i++) {
                    vector<double> row;
                    for (int j = 0; j < matrix.colNum; j++) {
                        double value = 0;
                        for (int k = 0; k < colNum; k++) {
                            value += values[i][k] * matrix.values[k][j];
                        }
                        row.push_back(value);
                    }
                    newValues.push_back(row);
                }
                newMatrix.values = newValues;
                return newMatrix;
            }
        }
        template <typename T>
        Matrix operator* (const T& number) const {
            if((std::is_same<T, int>::value || 
                is_same<T, double>::value || 
                is_same<T, float>::value) == 0) {
                cout << "Error: 2nd operand not a number!" << endl;
                return nullptr;
            }
            else {
                Matrix newMatrix(rowNum, colNum);
                for (int i = 0; i < rowNum; i++) {
                    for (int j = 0; j < colNum; j++) {
                        newMatrix.values[i][j] = (*this).values[i][j] * number;
                    }
                }
                return newMatrix;
            }
        }
        template <typename T>
        Matrix operator/ (const T& number) const {
            if((std::is_same<T, int>::value || 
                is_same<T, double>::value || 
                is_same<T, float>::value) == 0) {
                cout << "Error: 2nd operand not a number!" << endl;
                return nullptr;
            }
            else {
                Matrix newMatrix(rowNum, colNum);
                for (int i = 0; i < rowNum; i++) {
                    for (int j = 0; j < colNum; j++) {
                        newMatrix.values[i][j] = (*this).values[i][j] / number;
                    }
                }
                return newMatrix;
            }
        }
        
        // METHODS
        void setValues(vector<vector<double>> newValues) {
            values = newValues;
            rowNum = newValues.size();
            colNum = newValues[0].size();
        }
        vector<vector<double>> getValues() {
            return values;
        }

        void changeValue(int row, int col, double newValue) {
            if (row > rowNum || col > colNum) {
                cout << "Error: row or col number > matrix size" << endl;
            }
            else {
                values[row][col] = newValue;
            }
        }
        double getValue(int row, int col) const {
            if (row > rowNum || col > colNum) {
                cout << "Error: row or col number > matrix size" << endl;
                return 0;
            }
            else {
                return values[row][col];
            }
        }

        void setRowNumber(int newRowNum) {
            if (newRowNum < rowNum) {
                cout << "Error: new row number < current row number" << endl;
            }
            else {
                vector<double> newMatrixRow(colNum, 0);
                values.push_back(newMatrixRow);
                rowNum = newRowNum;
            }
        }
        int getRowNumber() const {
            return rowNum;
        }
        
        void setColNumber(int newColNum) {
            if (newColNum < colNum) {
                cout << "Error: new column number < col number" << endl;
            }
            else {
                colNum = newColNum;
                for (int i = 0; i < rowNum; i++) {
                    values.at(i).push_back(0);
                }
            }
        }
        int getColNumber() const {
            return colNum;
        }
        
        Matrix getColumn(int index) {
            Matrix result(this->getRowNumber(), 1);
            for (int i = 0; i < this->getRowNumber(); i++) {
                result.values[i][0] = this->values[i][index];
            }
            return result;
        }

        void switchRows(int rowOne, int rowTwo) {
            if (rowOne > rowNum || rowTwo > rowNum) {
                cout << "Error: row number > matrix size" << endl;
            }
            else {
                vector<double> temp = values[rowOne];
                values[rowOne] = values[rowTwo];
                values[rowTwo] = temp;
            }
        }
        void switchColumns(int colOne, int colTwo) {
            if (colOne > colNum || colTwo > colNum) {
                cout << "Error: column number > matrix size" << endl;
            }
            else {
                for (int i = 0; i < rowNum; i++) {
                    double temp = values[i][colOne];
                    values[i][colOne] = values[i][colTwo];
                    values[i][colTwo] = temp;
                }
            }
        }
        
        Matrix transpose() {
            vector<vector<double>> newValues;
            for (int i = 0; i < colNum; i++) {
                vector<double> row;
                for (int j = 0; j < rowNum; j++) {
                    row.push_back(values[j][i]);
                }
                newValues.push_back(row);
            }
            values = newValues;
            int temp = rowNum;
            rowNum = colNum;
            colNum = temp;
            return *this;
        }
        static Matrix identity(int rows, int columns) {
            Matrix I = Matrix(rows, columns);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    if(i == j) {
                        I.changeValue(i, j, 1);
                    }
                }
            }
            return I;
        }
        void LU_to_A(Matrix P) {
            int n = P.getRowNumber();
            Matrix L(n, n);
            Matrix U(n, n);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i == j) {
                        L.values[i][j] = 1;
                        U.values[i][j] = this->values[i][j];
                    }
                    else if (i > j) {
                        L.values[i][j] = this->values[i][j];
                    }
                    else {
                        U.values[i][j] = this->values[i][j];
                    }
                }
            }
            *this = (P * L) * U;
        }
        vector<Matrix> LUSplit() {
            vector<Matrix> result;
            Matrix pom = *this;

            int n = this->getRowNumber();
            Matrix L(n, n);
            Matrix U(n, n);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (abs(pom.values[i][j]) < EPSILON) {
                        pom.values[i][j] = 0;
                    }
                    if (i == j) {
                        L.values[i][j] = 1;
                        U.values[i][j] = pom.values[i][j];
                    }
                    else if (i > j) {
                        L.values[i][j] = pom.values[i][j];
                        U.values[i][j] = 0;
                    }
                    else {
                        U.values[i][j] = pom.values[i][j];
                        L.values[i][j] = 0;
                    }
                }
            }
            result.push_back(L);
            result.push_back(U);
            return result;
        }

        // METODE POTREBNE ZA LABORATORIJSKU VJEŽBU
        int LU_Decomposition() {
            int n = this->getRowNumber();
            if (this->getColNumber() != this->getRowNumber()) {
                cout << "Error: non-square Matrix" << endl;
                return 1;
            }

            // za i = 1 do n-1
            //     za j = i+1 do n
            //         A[j,i] /= A[i,i];
            //         za k = i+1 do n
            //             A[j,k] -= A[j,i] * A[i,k];

            for (int i = 0; i < n - 1; i++) {
                if (this->values[i][i] < EPSILON) {
                    cout << "Error: singularna matrica" << endl;
                    return 1;
                }
                for (int j = i + 1; j < n; j++) {
                    this->values[j][i] = this->values[j][i] / this->values[i][i];
                    for (int k = i + 1; k < n; k++) {
                        this->values[j][k] -= (this->values[j][i] * this->values[i][k]);
                    }
                }
            }
            return 0;
        }
        Matrix LUP_Decomposition() {
            int n = this->getRowNumber();
            int pivotIndex = -1;
            double pivot;

            vector<int> P;
            for (int i = 0; i < n; i++) {
                P.push_back(i);
            }

            // za i = 1 do n-1
            // pivot = i;
            for (int i = 0; i < n - 1; i++) {
                pivot = i;
            
                // za j = i+1 do n
                //     ako ( abs(A[P[j],i]) > abs(A[P[pivot],i) )
                //         pivot = j;
                for (int j = i + 1; j < n; j++) {
                    if (abs(this->getValue(P[j], i)) > abs(this->getValue(P[pivot], i))) {
                        pivot = j;
                    }
                }

                if (abs(this->values[pivot][i]) < EPSILON) {
                    cout << "Error: singularna matrica LUP" << endl;
                    exit(0);
                }

                // zamijeni(P[i],P[pivot]);
                if (pivot != i) {

                    int pom = P[i];
                    P[i] = P[pivot];
                    P[pivot] = pom;
                }

                // za j = i+1 do n
                //     A[P[j],i] /= A[P[i],i];
                //     za k = i+1 do n
                //         A[P[j],k] -= A[P[j],i] * A[P[i],k];
                for (int j = i + 1; j < n; j++) {
                    this->values[P[j]][i] = this->values[P[j]][i] / this->values[P[i]][i];
                    for (int k = i + 1; k < n; k++) {
                        this->values[P[j]][k] -= (this->values[P[j]][i] * this->values[P[i]][k]);
                    }
                }
            }

            Matrix newP(n, n);
            for (int i = 0; i < n; i++) {
                newP.values[i][P[i]] = 1.0;
            }
            *this = newP * (*this);
            return newP;
        }

        void forwardSupstitution(Matrix L) {
            // this->values[0][0] = this->values[0][0] y1 = b1
            // za i = 1 do n-1
            // za j = i+1 do n
            //     b[j] -= A[j,i] * b[i];

            int n = L.getRowNumber();
            for (int i = 0; i < n - 1; i++) {
                for (int j = i + 1; j < n; j++) {
                    this->values[j][0] = this->values[j][0] - (L.values[j][i] * this->values[i][0]);
                }
            }
        }
        int backwardSupstitution(Matrix U) {
            // cout << "U: " << endl;
            // U.toString();
            // za i = n do 1
            // b[i] /= A[i,i];
            // za j = 1 do i-1
            //     b[j] -= A[j,i] * b[i];

            int n = U.getColNumber();
            for (int i = n - 1; i >= 0; i--) {
                if (abs(U.values[i][i]) < EPSILON) {
                    cout << "Error matrica je singularna - U[i][j] = 0";
                    return 1;
                }
                this->values[i][0] = this->values[i][0] / U.values[i][i];
                for (int j = 0; j < i; j++) {
                    this->values[j][0] = this->values[j][0] - (U.values[j][i] * this->values[i][0]);
                } 
            }
            return 0;
        }

        static Matrix solveEquation(Matrix A, Matrix b) {
            Matrix P = A.LUP_Decomposition();
            Matrix y(b);
            y = P * y;
            y.forwardSupstitution(A);
            int error = y.backwardSupstitution(A);
            if (error) {
                exit(0);
            }
            for (int i = 0; i < A.getRowNumber(); i++) {
                if (abs(y.values[i][0]) < EPSILON) {
                    y.values[i][0] = 0;
                }
            }
            return y;
        }

        Matrix inverse() {
            int n = this->getRowNumber();
            Matrix P = this->LUP_Decomposition();
            vector<Matrix> pom = this->LUSplit();
            Matrix L = pom[0];
            Matrix U = pom[1];

            Matrix b = Matrix::identity(n, n);
            Matrix inverse(n, n);
            b = P * b;
            for (int i = 0; i < n; i++) {
                Matrix result = b.getColumn(i);
                result.forwardSupstitution(L);
                result.backwardSupstitution(U);
                for (int j = 0; j < n; j++) {
                    inverse.values[j][i] = result.values[j][0];
                }
            }
            return inverse;
        }

        double LUPdeterminant() {
            Matrix P = this->LUP_Decomposition();
            int n = P.getRowNumber();
            int detP;
            double detU = 1;
            int rowSwitch = 0;
            for (int i = 0; i < n; i++) {
                if (P.values[i][i] - 1 > EPSILON) {
                    for (int j = i+1; j < n; j++) {
                        if (P.values[j][i] - 1 < EPSILON) {
                            P.values[j][j] = 1;
                            break;
                        }
                    }
                    rowSwitch++;
                }
                detU *= this->values[i][i];
            }
            detU = detU < EPSILON ? 0 : detU;
            detP = rowSwitch % 2 == 0 ? 1 : -1;
            return detP * detU;
        }

        // PRINTING
        void toString() {
            for (int i = 0; i < rowNum; i++) {
                for (int j = 0; j < colNum; j++) {
                    cout << values[i][j] << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
        void toFile(string filename) {
            ofstream matrixFile(filename);
            for (int i = 0; i < rowNum; i++) {
                for (int j = 0; j < colNum; j++) {
                    matrixFile << values[i][j] << " ";
                }
                matrixFile << endl;
            }
            matrixFile.close();
        }

};

ostream& operator<< (ostream& os, const Matrix& matrix) {
    for (int i = 0; i < matrix.getRowNumber(); i++) {
        for (int j = 0; j < matrix.getColNumber(); j++) {
            os << matrix.getValue(i, j) << " ";
        }
        os << endl;
    }
    return os;
}

int main () {
    string base = filesystem::current_path().string() + "\\tasks\\";

    // 1. Kakva treba biti usporedba double varijabli kako bi uspoređivanje dalo očekivane rezultate?
    // Isprobajte operator == s elementima matrice kao necijelim brojevima, pomnožite i podijelite s
    // realnim brojem i usporedite s originalom.



    cout << "1. zadatak:" << endl;
    Matrix first_A(base + "1.txt");
    first_A.toString();
    Matrix first_B = first_A * 3.27;
    first_B = first_B / 3.27;
    first_B.toString();

    cout << ((first_A == first_B) ? "Matrice su jednake" : "Matrice nisu jednake") << endl;


    // 2. Riješite sustav zadan matricama u nastavku. Odredite može li se riješiti LU odnosno LUP
    // dekompozicijom:

    cout << "---------------------------------" << endl;
    cout << "2. zadatak:" << endl;
    cout << "LU dekompozicija :" << endl;
    Matrix A2(base + "2_A.txt");
    Matrix b2(base + "2_b.txt");
    Matrix A2_lu(A2);
    int error = A2_lu.LU_Decomposition();
    if (error) {
        cout << "Ne mozemo obaviti LU dekompoziciju" << endl;
    }
    else {
        Matrix x2_lu = Matrix::solveEquation(A2_lu, b2);
    }

    cout << "\nLUP dekompozicija :" << endl;
    Matrix x2 = Matrix::solveEquation(A2, b2);
    for (int i = 0; i < x2.getRowNumber(); i++) {
        cout << "x" << i << " = " << x2.getValue(i, 0) << endl;
    }


    // 3. Zadanu matricu rastavite na LU odnosno LUP. Ako je ovom matricom predstavljen sustav jednadžbi,
    // može li se sustav riješiti? (sami definirajte slobodni vektor)

    cout << "---------------------------------" << endl;
    cout << "3. zadatak:" << endl;

    Matrix A3_LU(base + "3_A.txt");
    Matrix A3_LUP(A3_LU);
    Matrix b3(base + "3_b.txt");

    int errorLU = A3_LU.LU_Decomposition();
    Matrix x3_LU(A3_LU.getRowNumber(), 1);
    if (error) {
        cout << "Ne mozemo obaviti LU dekompoziciju" << endl;
    }
    else {
        cout << "LU dekompozicija : " << endl;
        // A3_LU.toString();
        Matrix x3_lu = Matrix::solveEquation(A3_LU, b3);

        for (int i = 0; i < x2.getRowNumber(); i++) {
            cout << "x" << i << " = " << x2.getValue(i, 0) << endl;
        }
    }



    cout << "LUP dekompozicija : " << endl;
    cout << "matrica A je skoro singularna : " << endl;

    // cout << "A : " << endl;
    // A3_LUP.toString();
    // b3.toString();
    // Matrix x3_LUP = Matrix::solveEquation(A3_LUP, b3);
    // for (int i = 0; i < x3_LUP.getRowNumber(); i++) {
    //     cout << "x" << i << " = " << x3_LUP.getValue(i, 0) << endl;
    // }


    // 4. Zadani sustav riješite LU te LUP dekompozicijom. Objasnite razliku u rješenjima! (očituje se
    // prilikom uporabe double varijabli)

    cout << "---------------------------------" << endl;
    cout << "4. zadatak:" << endl;
    Matrix A4(base + "4_A.txt");
    Matrix b4(base + "4_b.txt");

    Matrix A4_LU(A4);
    Matrix A4_LUP(A4);

    int error_4_LU = A4_LU.LU_Decomposition();
    Matrix x4_LU(A4_LU.getRowNumber(), 1);
    if (error_4_LU) {
        cout << "Ne mozemo obaviti LU dekompoziciju" << endl;
    }
    else {
        cout << "LU dekompozicija : " << endl;
        // A4_LU.toString();
        Matrix x4_lu = Matrix::solveEquation(A4_LU, b4);

        for (int i = 0; i < x4_LU.getRowNumber(); i++) {
            cout << "x" << i << " = " << x4_LU.getValue(i, 0) << endl;
        }
    }

    cout << "LUP dekompozicija : " << endl;
    cout << "A : " << endl;
    A4_LUP.toString();
    b4.toString();
    Matrix x4_LUP = Matrix::solveEquation(A4_LUP, b4);
    for (int i = 0; i < x4_LUP.getRowNumber(); i++) {
        cout << "x" << i << " = " << x4_LUP.getValue(i, 0) << endl;
    }


    // 5. Zadani sustav riješite odgovarajućom metodom. Objasnite razliku između dobivenog i točnog
    // rješenja

    cout << "---------------------------------" << endl;
    cout << "5. zadatak:" << endl;

    Matrix A5(base + "5_A.txt");
    Matrix b5(base + "5_b.txt");
    Matrix x5 = Matrix::solveEquation(A5, b5);

    for (int i = 0; i < x5.getRowNumber(); i++) {
        cout << "x" << i << " = " << x5.getValue(i, 0) << endl;
    }

    // 6. Rješavanje sljedećeg sustava moglo bi zadati problema vašoj implementaciji. O čemu to ovisi? Kako
    // je moguće izbjeći ovaj problem, transformacijom zadanog sustava tako da rješenje ostane
    // nepromijenjeno? (Napomena: postavite vrijednost epsilona za ovaj primjer na6
    // 10−6 )

    cout << "---------------------------------" << endl;
    cout << "6. zadatak:" << endl;
    cout << "matrica je singularna:" << endl;

    Matrix A6(base + "6_A.txt");
    Matrix b6(base + "6_b.txt");
    // Matrix x6 = Matrix::solveEquation(A6, b6);

    // for (int i = 0; i < x6.getRowNumber(); i++) {
    //     cout << "x" << i << " = " << x6.getValue(i, 0) << endl;
    // }

    // 7. Korištenjem LUP dekompozicije izračunajte inverz zadane matrice te ispišite dobiveni rezultat:

    cout << "---------------------------------" << endl;
    cout << "7. zadatak:" << endl;
    cout << "MATRICA JE SINGULARNA !" << endl;

    // Matrix A7(base + "7_A.txt");
    // Matrix A7_copy(A7);
    // Matrix A7_inv = A7_copy.inverse();

    // cout << "A:" << endl;
    // A7.toString();
    // cout << "A^-1:" << endl;
    // A7_inv.toString();
    // cout << "A * A^-1:" << endl;
    // cout << (A7 * A7_inv) << endl;

    
    cout << "---------------------------------" << endl;
    cout << "8. zadatak:" << endl;

    Matrix A8(base + "8_A.txt");
    Matrix A8_copy(base + "8_A.txt");
    Matrix A8_inv = A8_copy.inverse();

    cout << "A:" << endl;
    A8.toString();
    cout << "A^-1:" << endl;
    A8_inv.toString();
    cout << "A * A^-1:" << endl;
    cout << A8 * A8_inv << endl;

    // 9. Korištenjem LUP dekompozicije izračunajte determinantu matrice:
    
    cout << "---------------------------------" << endl;
    cout << "9. zadatak:" << endl;

    Matrix A9(base + "8_A.txt");
    double detA = A9.LUPdeterminant();
    cout << "|A| :" << detA << endl;

    // 10. Korištenjem LUP dekompozicije izračunajte determinantu matrice:
    
    cout << "---------------------------------" << endl;
    cout << "10. zadatak:" << endl;

    Matrix A10(base + "10_A.txt");
    double detA10 = A10.LUPdeterminant();
    cout << "|A| :" << detA10 << endl;












    return 0;
    
}