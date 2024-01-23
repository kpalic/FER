#ifndef GRADESCENT_HPP
#define GRADESCENT_HPP

#include <vector>
#include <utility>
#include <cmath>
#include <iostream>
#include "../functions/IFunction.hpp"
#include "../functions/Functions.hpp"
#include "../algorithms/Algorithm.hpp"

using namespace std;

class GradientDescent : public Algorithm {
    public:
    int iterations = 0;

    vector<double> findMinimum(vector<double> startingPoint, IFunction& f, double epsilon, bool goldenCut) {
        vector<double> currentPoint = startingPoint;
        vector<double> gradient = f.getGradient(currentPoint);
        vector<double> nextPoint(f.getNumberOfParameters());

        double currentValue;
        double nextValue;
        
        if (goldenCut) {
            do {
                vector<double> normalizedGradient = normalizeGradient(gradient);
                double optimalLambda = getOptimalLambda(currentPoint, f, normalizedGradient);
                for (int i = 0; i < f.getNumberOfParameters(); i++) {
                    nextPoint[i] = currentPoint[i] - (optimalLambda * normalizedGradient[i]);
                }
                currentValue = f.getValue(currentPoint);
                nextValue = f.getValue(nextPoint);
                // cout << "----------------------------------------" << endl;
                // debug(f, currentPoint, gradient);
                gradient = f.getGradient(nextPoint);
                currentPoint = nextPoint;
                // cout << "----------------------------------------" << endl;
                // cout << "Gradient norm: " << getNorm(gradient) << endl;
                iterations++;
            } while (getNorm(gradient) > epsilon);
        }
        else {
            do {
                gradient = f.getGradient(currentPoint);
                for (int i = 0; i < f.getNumberOfParameters(); i++) {
                    nextPoint[i] = currentPoint[i] - gradient[i];
                }
                currentPoint = nextPoint;
                iterations++;
            } while (getNorm(gradient) > epsilon && iterations < 5000);
        }
        return currentPoint;
    }

    void debug(IFunction& f, vector<double> currentPoint, vector<double> gradient) {
        cout << "Current point: ";
        for (int i = 0; i < currentPoint.size(); i++) {
            cout << currentPoint[i] << " ";
        }
        cout << endl;
        cout << "Gradient: ";
        for (int i = 0; i < gradient.size(); i++) {
            cout << gradient[i] << " ";
        }
        cout << endl;
        cout << "Value: " << f.getValue(currentPoint) << endl;
        cout << endl;
    }

    double getOptimalLambda(vector<double> startingPoint, IFunction& f, vector<double> gradient) {
        // gradijent u ovoj funkciji je normaliziran
        double epsilon = 10e-7;

        // find unimodal interval
        double lambdaMin = 0;
        double lambdaMax = 1;
        vector<double> currentPoint = startingPoint;
        vector<double> nextPoint(f.getNumberOfParameters());
        for (int i = 0; i < f.getNumberOfParameters(); i++) {
            nextPoint[i] = currentPoint[i] - (lambdaMax * gradient[i]);
        }

        double valueNext = f.getValue(nextPoint);
        double valueCurrent = f.getValue(currentPoint);

        int iterations = 0;
        while (valueNext < valueCurrent) {
            lambdaMax *= 2;
            for (int i = 0; i < f.getNumberOfParameters(); i++) {
                nextPoint[i] = startingPoint[i] - (lambdaMax * gradient[i]);
            }
            valueCurrent = valueNext;
            valueNext = f.getValue(nextPoint);
            iterations++;
        }

        if (lambdaMax > 2) {
            lambdaMin = lambdaMax / 4; // 4 = 2^2 -> dvije prethodne tocke
        }

        double a = lambdaMin;
        double b = lambdaMax;
        double k = 0.5 * (sqrt(5) - 1);

        double c = b - k * (b - a);
        double d = a + k * (b - a);

        vector<double> cPoint(f.getNumberOfParameters());
        vector<double> dPoint(f.getNumberOfParameters());
        for (int i = 0; i < f.getNumberOfParameters(); i++) {
            cPoint[i] = currentPoint[i] - c * gradient[i];
            dPoint[i] = currentPoint[i] - d * gradient[i];
        }

        double valueC = f.getValue(cPoint);
        double valueD = f.getValue(dPoint);

        while (abs(b - a) > epsilon) {
            if (valueC < valueD) {
                b = d;
                d = c;
                valueD = valueC;
                c = b - k * (b - a);
                for (int i = 0; i < f.getNumberOfParameters(); i++) {
                    cPoint[i] = currentPoint[i] - c * gradient[i];
                }
                valueC = f.getValue(cPoint);
            }
            else {
                a = c;
                c = d;
                valueC = valueD;
                d = a + k * (b - a);
                for (int i = 0; i < f.getNumberOfParameters(); i++) {
                    dPoint[i] = currentPoint[i] - d * gradient[i];
                }
                valueD = f.getValue(dPoint);
            }
        }
        // cout << "Optimal lambda: " << (a + b) / 2.0 << endl;
        return (a + b) / 2.0;
    }
};

#endif