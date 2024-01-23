#include <iostream>
#include <vector>
#include <cmath>
#include <utility>

#include "functions/IFunction.hpp"
#include "functions/Functions.hpp"
#include "algorithms/GradientDescent.hpp"
#include "algorithms/Newton-Raphson.hpp"
#include "algorithms/Gauss-Newton.hpp"

#define EPSILON 10e-4

using namespace std;

void printResult(IFunction& f, vector<double> startingPoint) {
    GradientDescent gd;

    vector<double> minimum_GD = gd.findMinimum(startingPoint, f, EPSILON, true);

    cout << "Minimum: " << minimum_GD[0] << " " << minimum_GD[1] << endl;
    cout << "Iterations: " << gd.iterations << endl;

    Newton_Raphson nr;

    vector<double> minimum_NR = nr.findMinimum(startingPoint, f, EPSILON, true);

    cout << "Minimum: " << minimum_NR[0] << " " << minimum_NR[1] << endl;
    cout << "Iterations: " << nr.iterations << endl;

    Gauss_Newton gn;

    vector<double> minimum_GN = gn.findMinimum(startingPoint, f, EPSILON, true);

    cout << "Minimum: " << minimum_GN[0] << " " << minimum_GN[1] << endl;
    cout << "Iterations: " << gn.iterations << endl;
}

int main (void) {

    F1 f1;
    F2 f2; 
    F3 f3;

    cout << "F1" << endl;
    printResult(f1, {-1.9, 2});
    cout << "----------------------------------------" << endl;
    cout << "F2" << endl;
    printResult(f2, {0.1, 0.3});
    cout << "----------------------------------------" << endl;
    cout << "F3" << endl;
    printResult(f3, {0, 0});
    cout << "----------------------------------------" << endl;


    cout << "\n\n1. zadatak" << endl;
    GradientDescent gd;
    vector<double> minimum_GD = gd.findMinimum({0,0}, f3, EPSILON, true);

    cout << "Minimum: " << minimum_GD[0] << " " << minimum_GD[1] << endl;
    cout << "Iterations: " << gd.iterations << endl;
    cout << "----------------------------------------" << endl;

    minimum_GD = gd.findMinimum({0,0}, f3, EPSILON, false);
    cout << "Minimum: " << minimum_GD[0] << " " << minimum_GD[1] << endl;
    cout << "Iterations: " << gd.iterations << endl;
    cout << "----------------------------------------" << endl;

    cout << "\n\n2. zadatak" << endl;
    GradientDescent gd_1;
    GradientDescent gd_2;
    Newton_Raphson nr_1;
    Newton_Raphson nr_2;

    F1 f1_2;
    F2 f2_2;
    vector<double> minimum_GD_1_1 = gd_1.findMinimum({50,50}, f1_2, EPSILON, true);
    vector<double> minimum_GD_1_2 = gd_2.findMinimum({50,50}, f2_2, EPSILON, true);

    cout << "\nGradient descent" << endl;
    cout << "Minimum: " << minimum_GD_1_1[0] << " " << minimum_GD_1_1[1] << endl;
    cout << "Iterations: " << gd_1.iterations << endl;
    cout << "getValueCounter: " << f1_2.getValueCounter << endl;
    cout << "getGradientCounter: " << f1_2.getGradientCounter << endl;
    cout << "getHessianCounter: " << f1_2.getHessianCounter << endl;

    cout << "\nMinimum: " << minimum_GD_1_2[0] << " " << minimum_GD_1_2[1] << endl;
    cout << "Iterations: " << gd_2.iterations << endl;
    cout << "getValueCounter: " << f2_2.getValueCounter << endl;
    cout << "getGradientCounter: " << f2_2.getGradientCounter << endl;
    cout << "getHessianCounter: " << f2_2.getHessianCounter << endl;

    f1_2.resetCounters();
    f2_2.resetCounters();

    vector<double> minimum_NR_1_2 = nr_1.findMinimum({50,50}, f1_2, EPSILON, true);
    vector<double> minimum_NR_2_2 = nr_2.findMinimum({50,50}, f2_2, EPSILON, true);

    cout << "\nNewton-Raphson" << endl;
    cout << "Minimum: " << minimum_NR_1_2[0] << " " << minimum_NR_1_2[1] << endl;
    cout << "Iterations: " << nr_1.iterations << endl;
    cout << "getValueCounter: " << f1_2.getValueCounter << endl;
    cout << "getGradientCounter: " << f1_2.getGradientCounter << endl;
    cout << "getHessianCounter: " << f1_2.getHessianCounter << endl;

    cout << "\nMinimum: " << minimum_NR_2_2[0] << " " << minimum_NR_2_2[1] << endl;
    cout << "Iterations: " << nr_2.iterations << endl;
    cout << "getValueCounter: " << f2_2.getValueCounter << endl;
    cout << "getGradientCounter: " << f2_2.getGradientCounter << endl;
    cout << "getHessianCounter: " << f2_2.getHessianCounter << endl;
    cout << "----------------------------------------" << endl;

    cout << "\n\n3. zadatak" << endl;
    Newton_Raphson nr_1_3;
    Newton_Raphson nr_2_3;

    F4 f4_3a;
    F4 f4_3b;
    vector<double> minimum_NR_1_3 = nr_1_3.findMinimum({3,3}, f4_3a, EPSILON, false);
    vector<double> minimum_NR_2_3 = nr_2_3.findMinimum({1,2}, f4_3b, EPSILON, false);

    cout << "\nMinimum: " << minimum_NR_1_3[0] << " " << minimum_NR_1_3[1] << endl;
    cout << "Iterations: " << nr_1_3.iterations << endl;
    cout << "getValueCounter: " << f4_3a.getValueCounter << endl;
    cout << "getGradientCounter: " << f4_3a.getGradientCounter << endl;
    cout << "getHessianCounter: " << f4_3a.getHessianCounter << endl;

    cout << "\nMinimum: " << minimum_NR_2_3[0] << " " << minimum_NR_2_3[1] << endl;
    cout << "Iterations: " << nr_2_3.iterations << endl;
    cout << "getValueCounter: " << f4_3b.getValueCounter << endl;
    cout << "getGradientCounter: " << f4_3b.getGradientCounter << endl;
    cout << "getHessianCounter: " << f4_3b.getHessianCounter << endl;
    cout << "----------------------------------------" << endl;

    f4_3a.resetCounters();
    f4_3b.resetCounters();
    nr_1_3.iterations = 0;
    nr_2_3.iterations = 0;

    vector<double> minimum_NR_1_3_golden = nr_1_3.findMinimum({3,3}, f4_3a, EPSILON, true);
    vector<double> minimum_NR_2_3_golden = nr_2_3.findMinimum({1,2}, f4_3b, EPSILON, true);

    cout << "With golden cut" << endl;
    cout << "\nMinimum: " << minimum_NR_1_3_golden[0] << " " << minimum_NR_1_3_golden[1] << endl;
    cout << "Iterations: " << nr_1_3.iterations << endl;
    cout << "getValueCounter: " << f4_3a.getValueCounter << endl;
    cout << "getGradientCounter: " << f4_3a.getGradientCounter << endl;
    cout << "getHessianCounter: " << f4_3a.getHessianCounter << endl;

    cout << "\nMinimum: " << minimum_NR_2_3_golden[0] << " " << minimum_NR_2_3_golden[1] << endl;
    cout << "Iterations: " << nr_2_3.iterations << endl;
    cout << "getValueCounter: " << f4_3b.getValueCounter << endl;
    cout << "getGradientCounter: " << f4_3b.getGradientCounter << endl;
    cout << "getHessianCounter: " << f4_3b.getHessianCounter << endl;
    cout << "----------------------------------------" << endl;

    cout << "\n\n4. zadatak" << endl;
    Gauss_Newton gn_4_1;
    F1 f1_4;

    vector<double> minimum_GN_4_1 = gn_4_1.findMinimum({-1.9, 2}, f1_4, EPSILON, true);
    cout << "\nMinimum: " << minimum_GN_4_1[0] << " " << minimum_GN_4_1[1] << endl;
    cout << "Iterations: " << gn_4_1.iterations << endl;
        cout << "----------------------------------------" << endl;



    cout << "\n\n5. zadatak" << endl;
    Gauss_Newton gn_5_1;
    F5 f5_1;
    F5 f5_2;
    F5 f5_3;

    vector<double> minimum_GN_5_1 = gn_5_1.findMinimum({-2, 2}, f5_1, EPSILON, true);
    cout << "Starting Point: -2 2" << endl;
    cout << "Minimum: " << minimum_GN_5_1[0] << " " << minimum_GN_5_1[1] << endl;
    cout << "Iterations: " << gn_5_1.iterations << endl;

    gn_5_1.iterations = 0;
    vector<double> minimum_GN_5_2 = gn_5_1.findMinimum({2, 2}, f5_2, EPSILON, true);
    cout << "Starting Point: 2 2" << endl;
    cout << "Minimum: " << minimum_GN_5_2[0] << " " << minimum_GN_5_2[1] << endl;
    cout << "Iterations: " << gn_5_1.iterations << endl;

    gn_5_1.iterations = 0;
    vector<double> minimum_GN_5_3 = gn_5_1.findMinimum({2, -2}, f5_3, EPSILON, true);
    cout << "Starting Point: 2 -2" << endl;
    cout << "Minimum: " << minimum_GN_5_3[0] << " " << minimum_GN_5_3[1] << endl;
    cout << "Iterations: " << gn_5_1.iterations << endl;

    cout << "----------------------------------------" << endl;





    
    return 0;


}