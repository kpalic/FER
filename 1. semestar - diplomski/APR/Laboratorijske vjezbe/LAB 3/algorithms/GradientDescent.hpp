#ifndef GRADESCENT_HPP
#define GRADESCENT_HPP

#include <vector>
#include <utility>
#include <cmath>
#include <iostream>
#include "../functions/IFunction.hpp"
#include "../functions/Functions.hpp"

using namespace std;

class GradientDescent {
    public:

    vector<double> findMinimum(vector<double> startingPoint, IFunction& f, double epsilon, bool goldenCut) {
        vector<double> currentPoint = startingPoint;
        vector<double> gradient = f.getGradient(currentPoint);
        vector<double> nextPoint(f.getNumberOfParameters());
        
        if (goldenCut) {
            do {
                double optimalLambda = getOptimalLambda(currentPoint, f, gradient, epsilon);
                for (int i = 0; i < f.getNumberOfParameters(); i++) {
                    nextPoint[i] = currentPoint[i] - optimalLambda * gradient[i];
                }
                debug(f, currentPoint, gradient);
                gradient = f.getGradient(nextPoint);
                currentPoint = nextPoint;
            } while (getNorm(gradient) > epsilon);
        }
        else {
            do {
                for (int i = 0; i < f.getNumberOfParameters(); i++) {
                    nextPoint[i] = currentPoint[i] - gradient[i];
                }
                debug(f, currentPoint, gradient);
                gradient = f.getGradient(nextPoint);
                currentPoint = nextPoint;
            } while (getNorm(gradient) > epsilon);
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
        cout << "Value: " << f.getValue(currentPoint) << endl;
        cout << endl;
    }

    double getNorm(vector<double> gradient) {
        double norm = 0;
        for (int i = 0; i < gradient.size(); i++) {
            norm += pow(gradient[i], 2);
        }
        return sqrt(norm);
    }

    double getOptimalLambda(vector<double> currentPoint, IFunction& f, vector<double> gradient, double epsilon = 0.000001) {

        // find unimodal interval
        double lambdaMin = 0;
        double lambdaMax = 1;
        vector<double> nextPoint(f.getNumberOfParameters());
        for (int i = 0; i < f.getNumberOfParameters(); i++) {
            nextPoint[i] = currentPoint[i] - lambdaMax * gradient[i];
        }
        double valueNext = f.getValue(nextPoint);
        double valueCurrent = f.getValue(currentPoint);

        int iterations = 0;
        while (valueNext < valueCurrent) {
            if (iterations == 1)  {
                lambdaMin = 1;
            }
            else if (iterations > 1) {
                lambdaMin *= 2;
            }
            lambdaMax *= 2;

            for (int i = 0; i < f.getNumberOfParameters(); i++) {
                nextPoint[i] = currentPoint[i] - lambdaMax * gradient[i];
            }
            valueCurrent = valueNext;
            valueNext = f.getValue(nextPoint);
            iterations++;
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

        while ((b - a) > epsilon) {
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
        cout << "Optimal lambda: " << (a + b) / 2.0 << endl;
        return (a + b) / 2.0;
    }
};

#endif