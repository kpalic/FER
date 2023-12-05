#ifndef F1_Rosenbrock_HPP
#define F1_Rosenbrock_HPP

#include "IFunction.hpp"

using namespace std;

// Rosenbrock banana function
class F1_Rosenbrock : public IFunction {
    public:
    double getValue(vector<double> X) override {
        if (X.size() != 2) {
            throw "F1: invalid number of arguments";
        }
        return 100 * pow(X[1] - pow(X[0], 2), 2) + pow(1 - X[0], 2);
    }

    int getNumberOfParameters() override {
        return 2;
    }
};

#endif // F1