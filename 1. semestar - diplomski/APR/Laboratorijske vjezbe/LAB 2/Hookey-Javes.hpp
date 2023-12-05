#ifndef HOOKEYJAVES_HPP
#define HOOKEYJAVES_HPP

using namespace std;

#include "function/IFunction.hpp"

#include <vector>
#include <iostream>
#include <cmath>
#include <limits>

class HookeJeeves {

    public:
    int counter;
    HookeJeeves(double delta, double epsilon)
        : delta(delta), epsilon(epsilon) {}

    vector<double> optimize(IFunction& function, vector<double> startPoint, bool trace = true) {
        vector<double> x0 = startPoint;
        vector<double> xB = x0;
        vector<double> xP = x0;
        vector<double> Dx(function.getNumberOfParameters(), delta);
        vector<double> xN;
        counter = 0;
        int iteration = 0;
        while (!isStoppingCriterionMet(Dx)) {
            counter++;
            xN = explore(xP, Dx, function);

            if (trace) {
                cout << "Iteration " << iteration << ": ";
                printVector(xB);
                cout << " Start: ";
                printVector(xP);
                cout << " Result: ";
                printVector(xN);
                cout << " F(xB): " << function.getValue(xB);
                cout << " F(xP): " << function.getValue(xP);
                cout << " F(xN): " << function.getValue(xN) << endl;
            }

            if (function.getValue(xN) < function.getValue(xB)) {
                for (size_t i = 0; i < xP.size(); ++i) {
                    xP[i] = 2 * xN[i] - xB[i];
                }
                xB = xN;
            } else {
                for (double& d : Dx) {
                    d /= 2;
                }
                xP = xB;
            }

            iteration++;
        }

        return xB;
    }

private:
    double delta;
    double epsilon;

    bool isStoppingCriterionMet(const vector<double>& Dx) {
        for (double d : Dx) {
            if (d > epsilon) {
                return false;
            }
        }
        return true;
    }

    vector<double> explore(vector<double> xP, const vector<double>& Dx, IFunction& function) {
        vector<double> x = xP;
        for (size_t i = 0; i < x.size(); ++i) {
            double originalValue = x[i];

            x[i] += Dx[i];
            if (function.getValue(x) > function.getValue(xP)) {
                x[i] = originalValue - Dx[i];
                if (function.getValue(x) > function.getValue(xP)) {
                    x[i] = originalValue;
                }
            }
        }
        return x;
    }

    void printVector(const vector<double>& vec) {
        cout << "(";
        for (size_t i = 0; i < vec.size(); ++i) {
            cout << vec[i];
            if (i < vec.size() - 1) cout << ", ";
        }
        cout << ")";
    }
};



#endif