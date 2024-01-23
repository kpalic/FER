#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include <vector>

#include "../functions/IFunction.hpp"

using namespace std;

class F1 : public IFunction {
    public:
    double getValue(vector<double> X) {
        getValueCounter++;
        return 100 * pow(X[1] - pow(X[0], 2), 2) + pow(1 - X[0], 2);
    }

    vector<double> getGradient(vector<double> X) {
        vector<double> gradient;
        gradient.push_back(400 * pow(X[0], 3) - 400 * X[0] * X[1] + 2 * X[0] - 2);
        gradient.push_back(-200 * pow(X[0], 2) + 200 * X[1]);
        getGradientCounter++;
        return gradient;
    }

    int getNumberOfParameters() {
        return 2;
    }

    vector<vector<double>> getHessian(vector<double> X) {
        getHessianCounter++;
        vector<vector<double>> hessian;
        vector<double> row1;
        vector<double> row2;
        row1.push_back(1200 * pow(X[0], 2) - 400 * X[1] + 2);
        row1.push_back(-400 * X[0]);
        row2.push_back(-400 * X[0]);
        row2.push_back(200);
        hessian.push_back(row1);
        hessian.push_back(row2);
        return hessian;
    }

    vector<double> getG (vector<double> X) {
        vector<double> g;
        g.push_back(10 * X[1] - 10 * pow(X[0], 2));
        g.push_back(1 - X[0]);
        return g;
    }

    vector<vector<double>> getJacobijan (vector<double> X) {
        vector<vector<double>> jacobijan(2, vector<double>(2));
        jacobijan[0][0] = -20 * X[0];
        jacobijan[0][1] = 10;
        jacobijan[1][0] = -1;
        jacobijan[1][1] = 0;
        return jacobijan;
    }

    vector<vector<double>> getJacobijanInverse (vector<double> X) {
        vector<vector<double>> jacobijan = getJacobijan(X);
        vector<vector<double>> jacobijanInverse;
        vector<double> row1;
        vector<double> row2;
        double determinant = jacobijan[0][0] * jacobijan[1][1] - jacobijan[0][1] * jacobijan[1][0];

        if (abs(determinant) < 10e-6) {
            cerr << "Jacobijan matrix is singular!" << endl;
            exit(1);
        }

        row1.push_back(jacobijan[1][1] / determinant);
        row1.push_back(-jacobijan[0][1] / determinant);
        row2.push_back(-jacobijan[1][0] / determinant);
        row2.push_back(jacobijan[0][0] / determinant);
        jacobijanInverse.push_back(row1);
        jacobijanInverse.push_back(row2);
        return jacobijanInverse;
    }
};

class F2 : public IFunction {
    public:
    double getValue(vector<double> X) {
        getValueCounter++;
        return pow(X[0] - 4, 2) + 4 * pow(X[1] - 2, 2);
    }

    vector<double> getGradient(vector<double> X) {
        getGradientCounter++;
        vector<double> gradient;
        gradient.push_back(2 * (X[0] - 4));
        gradient.push_back(8 * (X[1] - 2));
        return gradient;
    }

    int getNumberOfParameters() {
        return 2;
    }

    vector<vector<double>> getHessian(vector<double> X) {
        getHessianCounter++;
        vector<double> row1;
        vector<double> row2;
        row1.push_back(2);
        row1.push_back(0);
        row2.push_back(0);
        row2.push_back(8);
        vector<vector<double>> hessian;
        hessian.push_back(row1);
        hessian.push_back(row2);
        return hessian;
    }

    vector<double> getG (vector<double> X) {
        vector<double> g;
        g.push_back(X[0] - 4);
        g.push_back(2*X[1] - 4);
        return g;
    }

    vector<vector<double>> getJacobijan (vector<double> X) {
        vector<vector<double>> jacobijan;
        vector<double> row1;
        vector<double> row2;
        row1.push_back(1);
        row1.push_back(0);
        row2.push_back(0);
        row2.push_back(2);
        jacobijan.push_back(row1);
        jacobijan.push_back(row2);
        return jacobijan;
    }

    vector<vector<double>> getJacobijanInverse(vector<double> X) {
        vector<vector<double>> jacobijanInverse(2, vector<double>(2));
        jacobijanInverse[0][0] = 1;
        jacobijanInverse[0][1] = 0;
        jacobijanInverse[1][0] = 0;
        jacobijanInverse[1][1] = 0.5;
        return jacobijanInverse;
    }
};

class F3 : public IFunction {
    public:
    double getValue(vector<double> X) {
        return pow(X[0] - 2, 2) + pow(X[1] + 3, 2);
    }

    vector<double> getGradient(vector<double> X) {
        vector<double> gradient;
        gradient.push_back(2 * (X[0] - 2));
        gradient.push_back(2 * (X[1] + 3));
        return gradient;
    }

    int getNumberOfParameters() {
        return 2;
    }

    vector<vector<double>> getHessian(vector<double> X) {
        vector<double> row1;
        vector<double> row2;
        row1.push_back(2);
        row1.push_back(0);
        row2.push_back(0);
        row2.push_back(2);
        vector<vector<double>> hessian;
        hessian.push_back(row1);
        hessian.push_back(row2);
        return hessian;
    }

    vector<double> getG (vector<double> X) {
        vector<double> g;
        g.push_back(X[0] - 2);
        g.push_back(X[1] + 3);
        return g;
    }

    vector<vector<double>> getJacobijan (vector<double> X) {
        vector<vector<double>> jacobijan(2, vector<double>(2));
        jacobijan[0][0] = 1;
        jacobijan[0][1] = 0;
        jacobijan[1][0] = 0;
        jacobijan[1][1] = 1;
        return jacobijan;
    }

    vector<vector<double>> getJacobijanInverse(vector<double> X) {
        vector<vector<double>> jacobijanInverse(2, vector<double>(2));
        jacobijanInverse[0][0] = 1;
        jacobijanInverse[0][1] = 0;
        jacobijanInverse[1][0] = 0;
        jacobijanInverse[1][1] = 1;
        return jacobijanInverse;
    }
};

class F4 : public IFunction {
    public:
    double getValue(vector<double> X) {
        getValueCounter++;
        return ((0.25 * pow(X[0], 4)) - pow(X[0], 2) + (2 * X[0]) + pow(X[1] - 1, 2));
    }
    
    vector<double> getGradient(vector<double> X) {
        getGradientCounter++;
        vector<double> gradient;
        gradient.push_back(pow(X[0], 3) - 2 * X[0] + 2);
        gradient.push_back(2 * (X[1] - 1));
        return gradient;
    }

    int getNumberOfParameters() {
        return 2;
    }

    vector<vector<double>> getHessian(vector<double> X) {
        getHessianCounter++;
        vector<double> row1;
        vector<double> row2;
        row1.push_back(3 * pow(X[0], 2) - 2);
        row1.push_back(0);
        row2.push_back(0);
        row2.push_back(2);
        vector<vector<double>> hessian;
        hessian.push_back(row1);
        hessian.push_back(row2);
        return hessian;
    }

    vector<double> getG (vector<double> X) {
        vector<double> g;
        g.push_back(X[0] - 1);
        g.push_back(X[1] - 1);
        return g;
    }

    vector<vector<double>> getJacobijan (vector<double> X) {
        vector<vector<double>> jacobijan;
        vector<double> row1;
        vector<double> row2;
        row1.push_back(1);
        row1.push_back(0);
        row2.push_back(0);
        row2.push_back(1);
        jacobijan.push_back(row1);
        jacobijan.push_back(row2);
        return jacobijan;
    }

    vector<vector<double>> getJacobijanInverse(vector<double> X) {
        vector<vector<double>> jacobijanInverse(2, vector<double>(2));
        jacobijanInverse[0][0] = 1;
        jacobijanInverse[0][1] = 0;
        jacobijanInverse[1][0] = 0;
        jacobijanInverse[1][1] = 1;
        return jacobijanInverse;
    }
};

class F5 : public IFunction {
    public:
    double getValue(vector<double> X) {
        getValueCounter++;
        return pow(X[1]*X[1] + X[0]*X[0] - 1, 2) + pow(X[1] - X[0]*X[0], 2);
    }

    int getNumberOfParameters() {
        return 2;
    }

    vector<double> getG (vector<double> X) {
        vector<double> g(2);
        g[0] = X[0] * X[0] + X[1] * X[1] - 1;
        g[1] = X[1] - X[0] * X[0];
        return g;
    }

    vector<vector<double>> getJacobijan (vector<double> X) {
        vector<vector<double>> jacobijan(2, vector<double>(2));
        jacobijan[0][0] = 2 * X[0];
        jacobijan[0][1] = 2 * X[1];
        jacobijan[1][0] = -2 * X[0];
        jacobijan[1][1] = 1;
        return jacobijan;
    }

    vector<double> getGradient(vector<double> X) {
        getGradientCounter++;
        vector<double> gradient;
        vector<vector<double>> jacobijanTranspose = transposeMatrix(getJacobijan(X));
        gradient = matrixMultiplication(jacobijanTranspose, getG(X));
        for (int i = 0; i < gradient.size(); i++) {
            gradient[i] *= 2;
        }
        return gradient;
    }

    //dont need this
    vector<vector<double>> getHessian(vector<double> X) {
        getHessianCounter++;
        vector<vector<double>> hessian;
        return hessian;
    }

    vector<vector<double>> getJacobijanInverse(vector<double> X) {
        vector<vector<double>> jacobijan = getJacobijan(X);
        vector<vector<double>> jacobijanInverse(2, vector<double>(2));
        double determinant = jacobijan[0][0] * jacobijan[1][1] - jacobijan[0][1] * jacobijan[1][0];
        if (abs(determinant) < 10e-9) {
            cout << "Jacobijan matrix is singular!" << endl;
            cerr << "Jacobijan matrix is singular!" << endl;
            exit(1);
        }
        jacobijanInverse[0][0] = jacobijan[1][1] / determinant;
        jacobijanInverse[0][1] = -jacobijan[0][1] / determinant;
        jacobijanInverse[1][0] = -jacobijan[1][0] / determinant;
        jacobijanInverse[1][1] = jacobijan[0][0] / determinant;

        return jacobijanInverse;
    }
};

class F6 : public IFunction {
    public:
    vector<int> t = {1,2,3,5,6,7};
    vector<int> y = {3,4,4,5,6,8};
    double getValue(vector<double> X) {
        getValueCounter++;
        double meanError = 0;
        for (int i = 0; i < t.size(); i++) {
            meanError += pow(y[i] - (X[0] * exp(X[1] * t[i]) + X[2]), 2);
        }
        return meanError / (1.0 * t.size());
    }

    
};

#endif