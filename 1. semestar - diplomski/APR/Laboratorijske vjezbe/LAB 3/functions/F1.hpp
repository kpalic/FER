#ifndef F1_HPP
#define F1_HPP

#include <vector>

#include "../functions/IFunction.hpp"

using namespace std;

// 100 ⋅ (𝑥ଶ − 𝑥ଵଶ)ଶ + (1 − 𝑥ଵ)ଶ 
class F1 : public IFunction {
    public:
    double getValue(vector<double> X) {
        return 100 * pow(X[0] - pow(X[1], 2), 2) + pow(1 - X[1], 2);
    }

    vector<double> getGradient(vector<double> X) {
        vector<double> gradient;
        
        return gradient;
    }

    int getNumberOfParameters() {
        return 2;
    }
};

#endif