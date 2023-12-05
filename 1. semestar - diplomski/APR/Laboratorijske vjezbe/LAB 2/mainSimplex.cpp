#include <iostream>
#include <vector>
#include <cmath>

#include "function/IFunction.hpp"
#include "function/F1_Rosenbrock.hpp"
#include "function/F2.hpp"
#include "function/F3.hpp"
#include "function/F4.hpp"
#include "function/F5.hpp"

#include "OptMethods/Simplex.hpp"

#define EPSILON 10e-6

using namespace std;

void printOptimum(vector<double> optimum, IFunction* f) {
    cout << "Optimum: [";
    for (int i = 0; i < optimum.size(); i++) {
        cout << optimum[i];
        if (i != optimum.size() - 1) {
            cout << ", ";
        }
    }
    cout << "]" << endl;
    cout << "Value: " << f->getValue(optimum) << endl;
}

int main () {
    int functionIndex;
    cout << "Choose function:" << endl;
    cout << "1. F1-Rosenbrock" << endl;
    cout << "2. F2" << endl;
    cout << "3. F3" << endl;
    cout << "4. F4-Jakobovic" << endl;
    cout << "5. F5-Schaffer" << endl;
    cin >> functionIndex;

    int maxIterations;
    cout << "Enter max number of iterations: ";
    cin >> maxIterations;

    cout << "Number of parameters: ";
    int n;
    cin >> n;

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
            break;
        default:
            break;
    }
    vector<double> optimum;
    Simplex simplex(1, 0.5, 2);
    cout << "\nNelder-Mead simplex" << endl;
    optimum = simplex.findOptimum(*f, startingPoint, 0.5, EPSILON, maxIterations);
    printOptimum(optimum, f);
}