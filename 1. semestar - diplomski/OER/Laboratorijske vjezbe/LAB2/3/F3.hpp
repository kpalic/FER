#include <vector>
#include <cmath>
#include <iostream>
#include "../1_2/IFunction.hpp"
#include "Matrix.hpp"

using namespace std;

class F3 : public IFunction {
    public:

    Matrix A;
    Matrix b;
    
    F3(Matrix A, Matrix b) {
        this->A = A;
        this->b = b;
    }

    int getNumberOfVariables() override {
        return A.n_cols;
    }    

    double getValue(vector<double> X) override {
        Matrix x = Matrix(X.size(), 1);
        for (int i = 0; i < X.size(); i++) {
            x.values[i][0] = X[i];
        }
        Matrix Ax_b = (A * x) - b;
        double result = Ax_b.norm();
        return result;
    }

    vector<double> getGradient(vector<double> X) override {
        Matrix x = Matrix(X.size(), 1);
        for (int i = 0; i < X.size(); i++) {
            x.values[i][0] = X[i];
        }
        vector<double> result;
        Matrix Ax_b = (A * x);
        Ax_b = Ax_b - b;
        Matrix ATAx_b = (A.transpose() * 2);
        ATAx_b = ATAx_b * Ax_b;
        for (int i = 0; i < ATAx_b.values.size(); i++) {
            result.push_back(ATAx_b.values[i][0]);
        }
        return result;
    }
    
};
