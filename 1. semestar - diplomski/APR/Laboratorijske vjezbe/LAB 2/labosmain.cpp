#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

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
#include "Hookey-Javes.hpp"

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

int main() {
    // 1. zadatak
    IFunction* f = new LabF1();
    vector<double> startingPoint;
    startingPoint.push_back(10);

    vector<double> optimum;
    IOptimizationMethod* method;

    cout << "1. zadatak" << endl;
    cout << "Coordinate axis search" << endl;
    method = new GoldenCut();
    optimum = method->findOptimum(*f, startingPoint, EPSILON, 1, 10000, true);
    printOptimum(optimum, f, method);


    cout << "\nNelder-Mead simplex" << endl;
    Simplex simplex(1, 0.5, 2);
    optimum = simplex.findOptimum(*f, startingPoint, 0.5, EPSILON, 1000);
    printOptimum(optimum, f);

    cout << "\nHooke-Jeeves" << endl;
    HookeJeeves hj(0.5, EPSILON);
    optimum = hj.optimize(*f, startingPoint);
    printOptimum(optimum, f);

    // 2. zadatak
    vector<IFunction*> functions = {new F1_Rosenbrock(), new F2(), new F3(), new F4()};
    vector<string> functionNames = {"F1", "F2", "F3", "F4"};
    startingPoint = {10, 10};
    vector<double> startingPoint3 = {2, 3, 7, 5, 8};

    cout << "Optimization Results:" << endl;
        const int width = 30;  // Width of each column

        cout << left << setw(width) << "Method"
            << setw(width) << "Function"
            << setw(60) << "Optimum"
            << setw(width) << "Value"
            << setw(width) << "Iterations" << endl;

    for (int i = 0; i < functions.size(); i++) {
        if (i == 2) {
            startingPoint = startingPoint3;
        }
        else {
            startingPoint = {10, 10};
        }
        IFunction* f = functions[i];
        string fName = functionNames[i];

        // Coordinate axis search


        IOptimizationMethod* method = new GoldenCut();
        vector<double> optimum = method->findOptimum(*f, startingPoint, EPSILON, 1, 1000, false);
        cout << setw(width) << "Coordinate Axis Search" << setw(width) << fName;
        string s1 = "";
        for (int a = 0; a < optimum.size(); a++) {
            s1 = s1 + to_string(optimum[a]) + " ";
        }

        cout << setw(60) << s1 << setw(width) << f->getValue(optimum) << setw(width) << method->counter << endl;

        // Nelder-Mead simplex
        Simplex simplex(1, 0.5, 2);
        optimum = simplex.findOptimum(*f, startingPoint, 0.5, EPSILON, 1000);
        cout << setw(width) << "Nelder-Mead Simplex" << setw(width) << fName;
        string s2 = "";
        for (int a = 0; a < optimum.size(); a++) {
            s2 = s2 + to_string(optimum[a]) + " ";
        }
        cout << setw(60) << s2 << setw(width) << f->getValue(optimum) << setw(width) << simplex.iteration << endl;

        // Hooke-Jeeves
        HookeJeeves hj(0.5, EPSILON);
        optimum = hj.optimize(*f, startingPoint, false);
        cout << setw(width) << "Hooke-Jeeves"<< setw(width) << fName;
        string s = "";
        for (int a = 0; a < optimum.size(); a++) {
            s = s + to_string(optimum[a]) + " ";
        }
        cout << setw(60) << s << setw(width) << f->getValue(optimum)  << setw(width) << hj.counter << endl;
    }


    // 3. zadatak

    cout << "\n3. zadatak" << endl;
    startingPoint3 = {5, 5};
    IFunction* f_zad5 = new F4;

    optimum = simplex.findOptimum(*f_zad5, startingPoint3, 0.5, EPSILON, 1000);
    cout << setw(width) << "Nelder-Mead Simplex" << setw(width) << "F5";
    string s2 = "";
    for (int a = 0; a < optimum.size(); a++) {
        s2 = s2 + to_string(optimum[a]) + " ";
    }
    cout << setw(60) << s2 << setw(width) << f_zad5->getValue(optimum) << setw(width) << simplex.iteration << endl;

    optimum = hj.optimize(*f_zad5, startingPoint3, false);
    cout << setw(width) << "Hooke-Jeeves"<< setw(width) << "F5";
        string s = "";
        for (int a = 0; a < optimum.size(); a++) {
            s = s + to_string(optimum[a]) + " ";
        }
    cout << setw(60) << s << setw(width) << f_zad5->getValue(optimum)  << setw(width) << hj.counter << endl;
    


    // 4. zadatak
    cout << "\n4. zadatak" << endl;
    vector<double> startingPoint4 = {0.5, 0.5};
    IFunction* f_zad4 = new F1_Rosenbrock();
    Simplex simp4(1, 0.5, 2);
    for (int i = 0; i < 20; i++) {
        vector<double> optimum4 = simp4.findOptimum(*f_zad4, startingPoint4, (double)i, EPSILON, 1000);
        printOptimum(optimum4, f_zad4);
    }
}   