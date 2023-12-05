#include <iostream>
#include <vector>
#include <cmath>

#include "function/IFunction.hpp"
#include "function/F1_Rosenbrock.hpp"
#include "function/F2.hpp"
#include "function/F3.hpp"
#include "function/F4.hpp"
#include "function/F5.hpp"
#include "function/labF1.hpp"

#include "OptMethods/IOptimizationMethods.hpp"
#include "OptMethods/GoldenCut.hpp"
#include "OptMethods/Simplex.hpp"

#define EPSILON 10e-6

using namespace std;

void printOptimum(vector<double> optimum, IFunction* f, IOptimizationMethod* method) {
    cout << "Optimum: [";
    for (int i = 0; i < optimum.size(); i++) {
        cout << optimum[i];
        if (i != optimum.size() - 1) {
            cout << ", ";
        }
    }
    cout << "]" << endl;
    cout << "Value: " << f->getValue(optimum) << endl;
    cout << "Number of iterations: " << method->counter << endl;
}

int main () {

    int option;
    cout << "Choose option:" << endl;
    cout << "1. Coordinate axis search" << endl;
    cout << "2. Nelder-Mead simplex" << endl;
    cin >> option;

    int functionIndex;
    cout << "Choose function:" << endl;
    cout << "1. F1-Rosenbrock" << endl;
    cout << "2. F2" << endl;
    cout << "3. F3" << endl;
    cout << "4. F4-Jakobovic" << endl;
    cout << "5. F5-Schaffer" << endl;
    cin >> functionIndex;

    cout << "Number of parameters: ";
    int n;
    cin >> n;

    cout << "Trace? (y/n): ";
    char traceAnswer;
    cin >> traceAnswer;

    cout << "Enter interval? (y/n): ";
    char intervalAnswer;
    cin >> intervalAnswer;
    vector<double> bounds;
    if (intervalAnswer == 'y') {
        cout << "Enter interval: ";
        double x;
        for (int i = 0; i < 2*n; i++) {
            cin >> x;
            bounds.push_back(x);
        }
    }

    int maxIterations;
    cout << "Enter max number of iterations: ";
    cin >> maxIterations;

    vector<double> startingPoint;
    cout << "Enter starting point? (y/n): ";
    char startingPointAnswer;
    cin >> startingPointAnswer;
    if (startingPointAnswer == 'y') {
        cout << "Enter starting point: ";
        double x;
        for (int i = 0; i < n; i++) {
            cin >> x;
            startingPoint.push_back(x);
        }
    }

    cout << "Option: " << option << endl;
    cout << "Function: " << functionIndex << endl;
    cout << "Max iterations: " << maxIterations << endl;
    cout << "Starting point: [";
    for (int i = 0; i < startingPoint.size(); i++) {
        cout << startingPoint[i];
        if (i != startingPoint.size() - 1) {
            cout << ", ";
        }
    }
    cout << "]" << endl;

    IFunction* f;
    switch (functionIndex)
    {
        case 1:
            f = new F1_Rosenbrock();
            if (startingPoint.size() == 0) {
                startingPoint.push_back(-1.9);
                startingPoint.push_back(2);
            }
            break;
        case 2:
            f = new F2();
            if (startingPoint.size() == 0) {
                startingPoint.push_back(0.1);
                startingPoint.push_back(0.3);
            }
            break;
        case 3:
            if (startingPoint.size() == 0) {
                startingPoint.push_back(0);
                startingPoint.push_back(0);
            }
            f = new F3(startingPoint.size());
            break;
        case 4:
            if (startingPoint.size() == 0) {
                startingPoint.push_back(5.1);
                startingPoint.push_back(1.1);
            }
            f = new F4();
            break;
        case 5:
            if (startingPoint.size() == 0) {
                startingPoint.push_back(1);
                startingPoint.push_back(1);
            }
            f = new F5(startingPoint.size());
        case 6: 
            if (startingPoint.size() == 0) {
                startingPoint.push_back(5);
            }
            f = new LabF1();
            break;
        default:
            break;
    }

    vector<double> optimum;
    double value;
    IOptimizationMethod* method;
    Simplex simplex(1,2,3);
    switch (option)
    {
        case 1:
            cout << "\nCoordinate axis search" << endl;
            if (intervalAnswer == 'y') {
                method = new GoldenCut(bounds);
            }
            else {
                method = new GoldenCut();
            }
            optimum = traceAnswer == 'y' ? method->findOptimum(*f, startingPoint, EPSILON, 1, maxIterations, true) 
                                        : method->findOptimum(*f, startingPoint, EPSILON, 1, maxIterations);
            printOptimum(optimum, f, method);
            break;
        case 2:
            break;  

        default:
            break;
    }

    return 0;
}

