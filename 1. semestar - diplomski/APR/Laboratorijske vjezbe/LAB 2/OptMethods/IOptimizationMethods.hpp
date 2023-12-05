#ifndef IOPTIMIZATIONMETHODS_HPP
#define IOPTIMIZATIONMETHODS_HPP

#include <vector>
#include "..\function\IFunction.hpp"

using namespace std;

class IOptimizationMethod {
    public:
    int counter = 0;
    virtual vector<double> findOptimum(IFunction& f,
                                       vector<double> startingPoint,
                                       double precision,
                                       double h,
                                       int maxIterations,
                                       bool trace = false) = 0;
};

#endif // IOPTIMIZATIONMETHODS_HPP