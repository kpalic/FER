#include <vector>
#include <iostream>
#include <cmath>

#define EPSILON2 10e-20

using namespace std;

class Matrix {
    public:
        vector<vector<double>> values;
        int n_rows;
        int n_cols;

        Matrix() {
            n_rows = 0;
            n_cols = 0;
        }

        Matrix(int n_rows, int n_cols) {
            this->n_rows = n_rows;
            this->n_cols = n_cols;
            values = vector<vector<double>>(n_rows, vector<double>(n_cols, 0));
        }

        Matrix operator+ (Matrix &m) {
            Matrix result;
            for (int i = 0; i < values.size(); i++) {
                vector<double> row;
                for (int j = 0; j < values[i].size(); j++) {
                    row.push_back(values[i][j] + m.values[i][j]);
                }
                result.values.push_back(row);
            }
            return result;
        }

        Matrix operator* (Matrix &m) {

            if (n_cols != m.n_rows) {
                cout << "Matrice nisu kompatibilne za mnozenje" << endl;
                return Matrix();
            }
            Matrix result(n_rows, m.n_cols);
            for (int i = 0; i < n_rows; i++) {
                for (int j = 0; j < m.n_cols; j++) {
                    double sum = 0;
                    for (int k = 0; k < n_rows; k++) {
                        sum += values[i][k] * m.values[k][j];
                    }
                    result.values[i][j] = sum;
                }
            }
            return result;
        }

        Matrix operator- (Matrix &m) {
            Matrix result(n_rows, n_cols);
            for (int i = 0; i < n_rows; i++) {
                for (int j = 0; j < n_cols; j++) {
                    if (abs(values[i][j] - m.values[i][j]) < EPSILON2) {
                        result.values[i][j] = 0;
                    }
                    else {
                        result.values[i][j] = values[i][j] - m.values[i][j];
                    }
                }
            }
            return result;
        }

        Matrix operator* (double d) {
            Matrix result(n_rows, n_cols);
            for (int i = 0; i < n_rows; i++) {
                for (int j = 0; j < n_cols; j++) {
                    result.values[i][j] = values[i][j] * d;
                }
            }
            return result;
        }

        Matrix transpose() {
            Matrix result = Matrix(n_cols, n_rows);
            for (int i = 0; i < n_rows; i++) {
                for (int j = 0; j < n_cols; j++) {
                    result.values[j][i] = values[i][j];
                }
            }
            return result;
        }

        double norm() {
            double result = 0;
            for (int i = 0; i < n_rows; i++) {
                for (int j = 0; j < n_cols; j++) {
                    result += values[i][j] * values[i][j];
                }
            }
            return sqrt(result);
        }

        void toString() {
            for (int i = 0; i < n_rows; i++) {
                for (int j = 0; j < n_cols; j++) {
                    cout << this->values[i][j] << " ";
                }
                cout << endl;
            }
        }

};