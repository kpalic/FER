#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include <vector>

#include "../functions/IFunction.hpp"

using namespace std;

class F1 : public IFunction {
    public:
    double getValue(vector<double> X) {
        return 100 * pow(X[0] - pow(X[1], 2), 2) + pow(1 - X[1], 2);
    }

    vector<double> getGradient(vector<double> X) {
        vector<double> gradient;
        gradient.push_back(400 * pow(X[0], 3) - 400 * X[0] * X[1] + 2 * X[0] - 2);
        gradient.push_back(-200 * pow(X[0], 2) + 200 * X[1]);
        return gradient;
    }

    int getNumberOfParameters() {
        return 2;
    }
};

class F2 : public IFunction {
    public:
    double getValue(vector<double> X) {
        return pow(X[0] - 4, 2) + 4 * pow(X[1] - 2, 2);
    }

    vector<double> getGradient(vector<double> X) {
        vector<double> gradient;
        gradient.push_back(2 * (X[0] - 4));
        gradient.push_back(8 * (X[1] - 2));
        return gradient;
    }

    int getNumberOfParameters() {
        return 2;
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
};

class F4 : public IFunction {
    public:
    double getValue(vector<double> X) {
        return ((0.25 * pow(X[0], 4)) - pow(X[0], 2) + (2 * X[0]) + pow(X[1] - 1, 2));
    }
    
    vector<double> getGradient(vector<double> X) {
        vector<double> gradient;
        gradient.push_back(pow(X[0], 3) - 2 * X[0] + 2);
        gradient.push_back(2 * (X[1] - 1));
        return gradient;
    }

    int getNumberOfParameters() {
        return 2;
    }
};

#endif