#include <iostream>
#include <vector>
#include "IFunction.hpp"

#define EPSILON 10e-12

using namespace std;

class NumOptAlgorithms {
    public:
    static vector<double> gradientDescent(IFunction& function, int maxIter, const vector<double> startPoint) {
        vector<double> currentPoint = startPoint;
        double minimum = LONG_MAX;
        int iterations = 0;
        vector<double> gradient = function.getGradient(currentPoint);
        double gradientSum;

        do {
            gradient = function.getGradient(currentPoint);
            double bestLambda = bestLearningRate(function, currentPoint);
            for (int i = 0; i < gradient.size(); i++) {
                currentPoint[i] = currentPoint[i] - (gradient[i] * bestLambda);
            }

            gradientSum = 0;
            for (int i = 0; i < gradient.size(); i++) {
                gradientSum += gradient[i];
            }
            cout << "Iteracija " << iterations + 1 << endl;
            for (int i = 0; i < currentPoint.size(); i++) {
                cout << currentPoint[i] << " ";
            }
            cout << endl;
            cout << "Gradient: " << gradientSum << endl;
            cout << "Value: " << function.getValue(currentPoint) << endl;
            iterations++;
        } while (iterations < maxIter && abs(gradientSum) > EPSILON);

        return currentPoint;
    }

    static double bestLearningRate(IFunction& function, vector<double> currentPoint) {
        
        vector<double> gradient = function.getGradient(currentPoint);   
        long double lambdaMin = 0;
        long double lambdaMax = 0.01;

        long double valueLambdaMin = function.getValue(currentPoint);

        vector<double> nextPoint(currentPoint.size());
        for (int i = 0; i < currentPoint.size(); i++) {
            nextPoint[i] = currentPoint[i] - (gradient[i] * lambdaMax);
        }
        double valueLambdaMax = function.getValue(nextPoint);

        bool expandInterval = false;
        do {
            expandInterval = false;
            double newLambdaMax = lambdaMax * 2;
            vector<double> nextPoint(currentPoint.size());
            for (int i = 0; i < currentPoint.size(); i++) {
                nextPoint[i] = currentPoint[i] - (gradient[i] * newLambdaMax);
            }
            double newValueLambdaMax = function.getValue(nextPoint);
            if (newValueLambdaMax < valueLambdaMax) {
                lambdaMax = newLambdaMax;
                valueLambdaMax = newValueLambdaMax;
                expandInterval = true;
            }
        } while (expandInterval == true);

        long double bestLambda = ((lambdaMax + lambdaMin) / 2) + lambdaMin;
        gradient = function.getGradient(currentPoint);

        long double gradientValue = 10e9;
        long double minGradientValue = 10e10;
        while (lambdaMax - lambdaMin > 10-6) {
            gradientValue = 0;
            vector<double> nextPoint(currentPoint.size());
            for (int i = 0; i < currentPoint.size(); i++) {
                nextPoint[i] = currentPoint[i] - (gradient[i] * bestLambda); 
            }
            vector<double> newGradient = function.getGradient(nextPoint);
            for (int j = 0; j < newGradient.size(); j++) {
                gradientValue += newGradient[j];
            }

            if (abs(gradientValue - minGradientValue) < EPSILON) {
            }
            else if (gradientValue > 0) {
                lambdaMax = bestLambda;
                bestLambda = ((lambdaMax - lambdaMin) / 2.0) + lambdaMin;
            }
            else {
                lambdaMin = bestLambda;
                bestLambda = ((lambdaMax - lambdaMin) / 2.0) + lambdaMin;
            }

            minGradientValue = gradientValue;
            cout << "Gradient = " << gradientValue << endl;
        }
        return bestLambda;
    }
};