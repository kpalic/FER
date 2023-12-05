#ifndef F2_HPP
#define F2_HPP

#include "IFunction.hpp"

using namespace std;

class F2 : public IFunction {
    public:
    double getValue(vector<double> X) override {
        if (X.size() != 2) {
            throw "F2: invalid number of arguments";
        }
        return pow(X[0] - 4, 2) + 4 * pow(X[1] - 2, 2);
    }

    int getNumberOfParameters() override {
        return 2;
    }
};

#endif // F2