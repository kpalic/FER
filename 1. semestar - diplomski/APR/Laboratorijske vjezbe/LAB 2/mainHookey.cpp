#include <iostream>
#include <vector>

#include "Hookey-Javes.hpp"
#include "function/F1_Rosenbrock.hpp"

using namespace std;

int main() {
    // Kreiranje instance Rosenbrockove funkcije
    F1_Rosenbrock rosenbrockFunction;
    // Parametri za Hooke-Jeeves algoritam
    double delta = 0.5;  // početni korak istraživanja
    double epsilon = 1e-6;  // preciznost za zaustavljanje

    // Kreiranje instance Hooke-Jeeves algoritma
    HookeJeeves hj(delta, epsilon);

    vector<double> startPoint = { 100, 100};
    vector<double> optimum = hj.optimize(rosenbrockFunction, startPoint);

    // Ispis rezultata
    cout << "Optimum found at: ";
    for (double value : optimum) {
        cout << value << " ";
    }
    cout << endl;

    return 0;
}
