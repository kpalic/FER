#ifndef NEWTON_RAPHSON_HPP
#define NEWTON_RAPHSON_HPP

#include <vector>
#include <cmath>
#include "../functions/IFunction.hpp"
#include "../algorithms/Algorithm.hpp"

using namespace std;

class Newton_Raphson : public Algorithm {
    public:
    int iterations = 0;

    vector<double> findMinimum (vector<double> startingPoint, IFunction& f, double epsilon, bool goldenCut) {
        vector<double> currentPoint = startingPoint;
        vector<double> gradient(f.getNumberOfParameters());
        vector<vector<double>> hessianInverse(f.getNumberOfParameters(), vector<double>(f.getNumberOfParameters()));
        vector<double> nextPoint(f.getNumberOfParameters());

        double currentValue = f.getValue(currentPoint);
        double nextValue;
        vector<double> direction(2);

        if (goldenCut) {
            do {
                gradient = f.getGradient(currentPoint);
                hessianInverse = f.getHessianMatrixInverse(currentPoint);
                direction = f.matrixMultiplication(hessianInverse, gradient);
                vector<double> normalizedDirection = normalizeGradient(direction);
                double optimalLambda = getOptimalLambda(currentPoint, f, normalizedDirection);
                for (int i = 0; i < f.getNumberOfParameters(); i++) {
                    nextPoint[i] = currentPoint[i] - (optimalLambda * normalizedDirection[i]);
                }
                currentPoint = nextPoint;
                currentValue = f.getValue(currentPoint);
                nextValue = f.getValue(nextPoint);
                iterations++;
            } while (getNorm(gradient) > 0.01);
        }
        else {
            do {
                gradient = f.getGradient(currentPoint);
                hessianInverse = f.getHessianMatrixInverse(currentPoint);
                direction = f.matrixMultiplication(hessianInverse, gradient);
                for (int i = 0; i < f.getNumberOfParameters(); i++) {
                    nextPoint[i] = currentPoint[i] - direction[i];
                }
                currentPoint = nextPoint;
                iterations++;
            } while (getNorm(gradient) > 0.0001 && iterations < 5000);
        }
        return currentPoint;
    }

    double getOptimalLambda(vector<double> startingPoint, IFunction& f, vector<double> direction) {
        // gradijent u ovoj funkciji je normaliziran
        double epsilon = 10e-5;

        // find unimodal interval
        double lambdaMin = -1;
        double lambdaMax = 1;

        vector<double> minPoint(f.getNumberOfParameters());
        vector<double> maxPoint(f.getNumberOfParameters());
        for (int i = 0; i < f.getNumberOfParameters(); i++) {
            minPoint[i] = startingPoint[i] - lambdaMin * direction[i];
            maxPoint[i] = startingPoint[i] - lambdaMax * direction[i];
        }

        double minPointValue = f.getValue(minPoint);
        double maxPointValue = f.getValue(maxPoint);

        if (minPointValue > maxPointValue) {
            double temp = maxPointValue;
            do {
                temp = maxPointValue;
                lambdaMax *= 2;
                for (int i = 0; i < f.getNumberOfParameters(); i++) {
                    maxPoint[i] = startingPoint[i] - lambdaMax * direction[i];
                }
                maxPointValue = f.getValue(maxPoint);
            } while (temp > maxPointValue);

            if (lambdaMax > 4) {
                lambdaMin = lambdaMax / 4;
            }
        }
        else {
            double temp = minPointValue;
            do {
                temp = minPointValue;
                lambdaMin *= 2;
                for (int i = 0; i < f.getNumberOfParameters(); i++) {
                    minPoint[i] = startingPoint[i] - lambdaMin * direction[i];
                }
                minPointValue = f.getValue(minPoint);
            } while (temp > minPointValue);

            if (lambdaMin < -4) {
                lambdaMax = lambdaMin / 4;
            }
        }

        vector<double> currentPoint = startingPoint;

        double a = lambdaMin;
        double b = lambdaMax;
        double k = 0.5 * (sqrt(5) - 1);

        double c = b - k * (b - a);
        double d = a + k * (b - a);

        vector<double> cPoint(f.getNumberOfParameters());
        vector<double> dPoint(f.getNumberOfParameters());
        for (int i = 0; i < f.getNumberOfParameters(); i++) {
            cPoint[i] = currentPoint[i] - c * direction[i];
            dPoint[i] = currentPoint[i] - d * direction[i];
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
                    cPoint[i] = currentPoint[i] - c * direction[i];
                }
                valueC = f.getValue(cPoint);
            }
            else {
                a = c;
                c = d;
                valueC = valueD;
                d = a + k * (b - a);
                for (int i = 0; i < f.getNumberOfParameters(); i++) {
                    dPoint[i] = currentPoint[i] - d * direction[i];
                }
                valueD = f.getValue(dPoint);
            }
        }
        // cout << "Optimal lambda: " << (a + b) / 2.0 << endl;
        return (a + b) / 2.0;
    }
};

#endif