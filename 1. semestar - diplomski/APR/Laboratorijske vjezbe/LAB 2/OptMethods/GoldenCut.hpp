#ifndef GOLDENCUT_HPP
#define GOLDENCUT_HPP

#include <vector>
#include <utility>
#include <cmath>

#include "../function/IFunction.hpp"
#include "IOptimizationMethods.hpp"

using namespace std;

class GoldenCut : public IOptimizationMethod{
    private:
        vector<double> bounds;
        bool intervalGiven = false;
    public:

    GoldenCut(vector<double> bounds) {
        this->bounds = bounds;
        this->intervalGiven = true;
    }

    GoldenCut() {
        this->intervalGiven = false;
    }

    static double lOneNorm(vector<double> a, vector<double> b) {
        double sum = 0;
        for (int i = 0; i < a.size(); i++) {
            sum += abs(a[i] - b[i]);
        }
        return sum;
    }

    vector<double> findOptimum (IFunction& f,
                                vector<double> startingPoint,
                                double precision,
                                double h,
                                int maxIterations,
                                bool trace = false) override
    {

        int n = f.getNumberOfParameters();
        double valueLast = f.getValue(startingPoint);

        double valueCurrent;
        vector<double> a = startingPoint;
       
        do {
            valueLast = valueCurrent;
            for (int i = 0; i < n; i++) {
                pair<double, double> interval;
                if (intervalGiven) {
                    interval.first = bounds[2 * i];
                    interval.second = bounds[2 * i + 1];
                }
                else {
                    interval = IFunction::unimodalInterval(a, i, h, f);
                }
                a = findOptimumOneVariable(f, a, interval, i, precision, trace);
            }
            valueCurrent = f.getValue(a);

        } while (abs(valueCurrent - valueLast) > precision && counter < maxIterations);
        return a;
    }

    void getTrace(double a, double b, double c, double d,
                  double fa, double fb, double fc, double fd) {
        
        cout << "a = " << a << " c = " << c << " d = " << d << " b = " << b << endl;
        cout << "fa = " << fa << " fc = " << fc << " fd = " << fd << " fb = " << fb << endl;
        cout << "-----------------------------------------" << endl;
    }


    vector<double> findOptimumOneVariable(IFunction& f,
                                          vector<double> startingPoint,
                                          pair<double, double> interval,
                                          int variableIndex,
                                          double precision,
                                          bool trace = false)
    {
        counter++;
        double k = 0.5 * (sqrt(5) - 1);

        vector<double> A = startingPoint;
        vector<double> B = startingPoint;
        vector<double> C = startingPoint;
        vector<double> D = startingPoint;
        
        double a = interval.first;
        double b = interval.second;
        double c = b - k * (b - a);
        double d = a + k * (b - a);
        
        A[variableIndex] = a;
        double fa = f.getValue(A);
        B[variableIndex] = b;
        double fb = f.getValue(B);
        C[variableIndex] = c;
        double fc = f.getValue(C);
        D[variableIndex] = d;
        double fd = f.getValue(D);

        if (trace) {
            getTrace(a, b, c, d, fa, fb, fc, fd);
        }

        while (b - a > precision) {
            if (fc > fd) {
                A = C;
                a = c;
                fa = fc;
                C = D;
                c = d;
                fc = fd;
                d = a + k * (b - a);
                D[variableIndex] = d;
                fd = f.getValue(D);
            }
            else {
                B = D;
                b = d;
                fb = fd;
                D = C;
                d = c;
                fd = fc;
                c = b - k * (b - a);
                C[variableIndex] = c;
                fc = f.getValue(C);
            }
        }
        if (trace) {
            getTrace(a, b, c, d, fa, fb, fc, fd);
        }
        double newMin = (a + b) / 2;
        A[variableIndex] = newMin;
        return A;
    }                                    
};

#endif // GOLDENCUT_HPP