#include <vector>
#include <cmath>
#include <iostream>
#include "IFunction.hpp"
#include "../3/Matrix.hpp"

using namespace std;

class F4 : public IFunction {
    public:

    Matrix A;
    Matrix b;
    
    F4(Matrix A, Matrix b) {
        this->A = A;
        this->b = b;
    }

    int getNumberOfVariables() override {
        return A.n_cols;
    }    

    long double getValue(vector<double> coefs) override { //value of loss function
        return meanSquaredError(coefs);
    }

    double meanSquaredError(vector<double> coefs) {
        double result = 0;
        for (int i = 0; i < A.n_rows; i++) {
            result += getSquaredError(coefs, A.values[i], b.values[i][0]);
        }
        result /= (double)A.n_rows;
        return result;
    }

    double getSquaredError(vector<double> coefs, vector<double> X, double y) {
        double result = pow(f_x(coefs, X) - y, 2);
        return result;
    }

    double f_x(vector<double> coefs, vector<double> X) {

        // cout << "X: " << X[0] << " " << X[1] << " " << X[2] << " " << X[3] << " " << X[4] << endl;
        // cout << "Coef: " << coefs[0] << " " << coefs[1] << " " << coefs[2] << " " << coefs[3] << " " << coefs[4] << " " << coefs[5] << endl;

        double y = coefs[0] * X[0] +
                   coefs[1] * pow(X[0], 3) * X[1] +
                   coefs[2] * exp(coefs[3] * X[2]) * (1 + cos(coefs[4] * X[3])) +
                   coefs[5] * X[3] * pow(X[4], 2);
        // cout << "f_x: " << y << endl;
        return y;
    }

    double getPartialDerivative(vector<double> coef, vector<double> X, double y, int i) {
        double result = 0;
        double error = f_x(coef, X) - y;
        // cout << "y: " << y << endl;
        // cout << "f_x: " << f_x(coef, X) << endl;
        // cout << "Error: " << error << endl;

        
        switch (i) {
            case 0:
                result = error * X[0];
                break;
            case 1:
                result = error * (pow(X[0], 3) * X[1]);
                break;
            case 2:
                result = error * (exp(coef[3] * X[2]) * (1 + cos(coef[4] * X[3])));
                break;
            case 3:
                result = error * (coef[2] * X[2] * exp(coef[3] * X[2]) * (1 + cos(coef[4] * X[3])));  
                break;
            case 4:
                result = error * (-coef[2] * X[3] * exp(coef[3] * X[2]) * sin(coef[4] * X[3]));
                break;
            case 5:
                result = error * (X[3] * pow(X[4], 2));
                break;    
        }
        // cout << "Partial derivative " << i << ": " << result << endl;
        return result;
    }

    vector<double> getGradient(vector<double> coefs) override {

        vector<double> result(coefs.size());
        for (int i = 0; i < result.size(); i++) {
            long double partialDerivative = 0;
            for (int j = 0; j < A.n_rows; j++) {
                partialDerivative += getPartialDerivative(coefs, A.values[j], b.values[j][0], i);
            }
            // cout << "Partial derivative sum " << i << ": " << partialDerivative << endl;
            result[i] = partialDerivative;
        }
        long double norm = 0;
        for (int i = 0; i < result.size(); i++) {
            norm += pow(result[i], 2);
        }
        norm = sqrt(norm);
        for (int i = 0; i < result.size(); i++) {
            result[i] /= norm;
        }
        return result;
    }
    
};