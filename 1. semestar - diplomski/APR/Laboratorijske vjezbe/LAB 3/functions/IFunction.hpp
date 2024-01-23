#ifndef IFUNCTION_HPP
#define IFUNCTION_HPP

#include <vector>
#include <cmath>
#include <utility>

using namespace std;

class IFunction {
    public:
    int getValueCounter = 0;
    int getGradientCounter = 0;
    int getHessianCounter = 0;
    virtual double getValue(vector<double> X) = 0;
    virtual int getNumberOfParameters() = 0;
    virtual vector<double> getGradient(vector<double> X) = 0;
    virtual vector<vector<double>> getHessian(vector<double> X) = 0;
    virtual vector<double> getG (vector<double> X) = 0;
    virtual vector<vector<double>> getJacobijanInverse (vector<double> X) = 0;

    vector<vector<double>> transposeMatrix(vector<vector<double>> matrix) {
        vector<vector<double>> transposedMatrix(matrix[0].size(), vector<double>(matrix.size()));
        for (int i = 0; i < matrix.size(); i++) {
            for (int j = 0; j < matrix[0].size(); j++){
                transposedMatrix[j][i] = matrix[i][j];
            }
        }
        return transposedMatrix;
    }

    vector<vector<double>> getHessianMatrixInverse (vector<double> X) {
        vector<vector<double>> hessian = getHessian(X);
        double determinant = hessian[0][0] * hessian[1][1] - hessian[0][1] * hessian[1][0];
        if (abs(determinant) < 10e-6) {
            cerr << "Hessian matrix is singular!" << endl;
            exit(1);
        }
        vector<vector<double>> hessianInverse(hessian.size(), vector<double>(hessian[0].size()));
        hessianInverse[0][0] = hessian[1][1] / determinant;
        hessianInverse[0][1] = -hessian[0][1] / determinant;
        hessianInverse[1][0] = -hessian[1][0] / determinant;
        hessianInverse[1][1] = hessian[0][0] / determinant;
        
        return hessianInverse;
    }

    vector<double> matrixMultiplication(vector<vector<double>> matrix1, vector<double> matrix2) {
        vector<double> result(matrix1.size());
        for (int i = 0; i < matrix1.size(); i++) {
            for (int j = 0; j < matrix2.size(); j++) {
                result[i] += matrix1[i][j] * matrix2[j];
            }
        }
        return result;
    }

    vector<vector<double>> matrixMultiplication2(vector<vector<double>> matrix1, vector<vector<double>> matrix2) {
        vector<vector<double>> result(matrix1.size(), vector<double>(matrix2[0].size()));
        for (int i = 0; i < matrix1.size(); i++) {
            for (int j = 0; j < matrix2[0].size(); j++) {
                for (int k = 0; k < matrix2.size(); k++) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
        return result;
    }

    void resetCounters() {
        getValueCounter = 0;
        getGradientCounter = 0;
        getHessianCounter = 0;
    }
};

#endif // IFUNCTION_HPP