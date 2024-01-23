#ifndef GAUSS_NEWTON_HPP
#define GAUSS_NEWTON_HPP

#include <vector>
#include "../algorithms/Algorithm.hpp"
#include "../functions/IFunction.hpp"

using namespace std;

class Gauss_Newton : public Algorithm {
    public:
    int iterations = 0;
    double optimalLambda;
    vector<double> deltaX;

    vector<double> findMinimum(vector<double> startingPoint, IFunction& f, double epsilon, bool goldenCut) {

        vector<double> currentPoint = startingPoint;
        vector<double> nextPoint(f.getNumberOfParameters());
        do {
            vector<vector<double>> JacobijanInverse = f.getJacobijanInverse(currentPoint);
            vector<double> g = f.getG(currentPoint);
            deltaX = matrixMultiplication(JacobijanInverse, g);
            optimalLambda = getOptimalLambda(currentPoint, f, deltaX);
            for (int i = 0; i < f.getNumberOfParameters(); i++) {
                nextPoint[i] = currentPoint[i] - (optimalLambda * deltaX[i]);
            }
            currentPoint = nextPoint;
            iterations++;
        } while (abs(optimalLambda * (deltaX[0] + deltaX[1])) > epsilon && iterations < 5000);
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

    vector<double> matrixMultiplication(vector<vector<double>> matrix1, vector<double> matrix2) {
        vector<double> result(matrix1.size());
        for (int i = 0; i < matrix1.size(); i++) {
            for (int j = 0; j < matrix2.size(); j++) {
                result[i] += matrix1[i][j] * matrix2[j];
            }
        }
        return result;
    }

    vector<vector<double>> matrixMultiplication2(vector<vector<double>> matrix1, vector<vector<double>> matrix2) {
        vector<vector<double>> result(matrix1.size(), vector<double>(matrix2[0].size()));
        for (int i = 0; i < matrix1.size(); i++) {
            for (int j = 0; j < matrix2[0].size(); j++) {
                for (int k = 0; k < matrix2.size(); k++) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
        return result;
    }
};

#endif