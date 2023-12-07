#include <iostream>
#include <vector>
#include <cmath>
#include <utility>

#include "functions/IFunction.hpp"
#include "functions/Functions.hpp"
#include "algorithms/GradientDescent.hpp"

#define EPSILON 0.000001

using namespace std;

int main (void) {

    F1 f1;
    F2 f2;
    F3 f3;

    GradientDescent gd;

    vector<double> startingPoint;
    startingPoint.push_back(100);
    startingPoint.push_back(-100);

    vector<double> minimum = gd.findMinimum(startingPoint, f1, EPSILON, true);

    cout << "Minimum: " << minimum[0] << " " << minimum[1] << endl;


    minimum = gd.findMinimum(startingPoint, f1, EPSILON, false);
    cout << "Minimum: " << minimum[0] << " " << minimum[1] << endl;

    return 0;


}