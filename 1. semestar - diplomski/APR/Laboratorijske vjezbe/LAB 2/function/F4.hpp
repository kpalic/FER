#ifndef F4_HPP
#define F4_HPP

#include "IFunction.hpp"

using namespace std;

class F4 : public IFunction {

    public:
        double getValue(vector<double> X) override {
            if (X.size() != 2) {
                throw "F4: invalid number of arguments";
            }
            return (abs((X[0] - X[1]) * (X[0] + X[1])) + sqrt(pow(X[0], 2) + pow(X[1], 2)));
        }

        int getNumberOfParameters() override {
            return 2;
        }

};

#endif // F4