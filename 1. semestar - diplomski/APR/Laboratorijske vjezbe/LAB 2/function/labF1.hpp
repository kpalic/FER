#ifndef LABF1_HPP
#define LABF1_HPP

#include "IFunction.hpp"

using namespace std;


class LabF1 : public IFunction {
    public:
    double getValue(vector<double> X) override {
        if (X.size() != 1) {
            throw "LabF1: invalid number of arguments";
        }
        return ((X[0] - 3) * (X[0] - 3));
    }

    int getNumberOfParameters() override {
        return 1;
    }
};

#endif