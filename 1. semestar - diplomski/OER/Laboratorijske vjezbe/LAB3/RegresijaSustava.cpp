#include <iostream>
#include <filesystem>
#include <cmath>
#include <fstream>
#include "IFunction.hpp"
#include "SimulatedAnnealing.hpp"
#include "Matrix.hpp"
#include "F4.hpp"

using namespace std;

int main (int argc, char* argv[]) {    

    if (argc < 4) {
        cout << "Not enough arguments" << endl;
        return 1;
    }

    vector<double> startingPoint(6);
    if (argc > 4) {
        for (int i = 0; i < 6; i++) {
            startingPoint[i] = stod(argv[4 + i]);
        }
    }
    else {
        startingPoint = {1,1,1,1,1,1};
    }

    bool maximum = stoi(argv[1]);
    cout << "Maximum: " << maximum << endl;
    int maxIterations = stoi(argv[2]);
    string filename = argv[3];

    string base = filesystem::current_path().string();
    string fullPath = base + "\\" + filename;

    ifstream file(fullPath);
    cout << base << "\\" << filename << endl;
    if (!file.is_open()) {
        cout << "Neispravan naziv datoteke" << endl;
        return 1;
    }

    Matrix A(20, 5);
    Matrix b(20, 1);

    int row = 0;
    while (!file.eof()) {
        string line;
        getline(file, line);

        stringstream ss(line);
        string value;
        for (int i = 0; i < 5; i++) {
            ss >> value;
            A.values[row][i] = stod(value);
        }
        ss >> value;
        b.values[row][0] = stod(value);
        row++;
    }

    F4 function(A, b);

    vector<vector<double>> startingPoints;
    for (int i = 0; i < 10; i++) {
        vector<double> point(6);
        for (int j = 0; j < 6; j++) {
            point[j] = rand() % 100;
        }
        point[3] = (int)point[3] % 10;
        point[4] = fmod(point[4], 2 * acos(-1));
        startingPoints.push_back(point);
    }
    double bestT = 100;
    double bestAlpha = 0.95;
    double bestSigma = 1;
    double bestError = 10e50;


    // for (double c = 0.01; c < 10; c *= 1.5) {
    //     double error = 0;
    //     for (int start = 0; start < startingPoints.size(); start++) {
    //         vector<double> point = startingPoints[start];
    //         vector<double> result = SimulatedAnnealing::OptimumFinder(function, point, maximum, maxIterations, bestT, bestAlpha, c);
    //         error += function.getValue(result);
    //         cout << "T = " << bestT << " alpha = " << bestAlpha << " sigma = " << c << " error = " << error << endl;
            
    //     }
    //     if (error < bestError) {
    //         bestSigma = c;
    //         bestError = error;
    //     }
    //     cout << "----------------------------------------" << endl;
    //     cout << " T = " << bestT << " alpha = " << bestAlpha << " sigma = " << bestSigma << " error = " << bestError << endl;
    //     cout << "----------------------------------------" << endl;
    // }

    // for (double b = 0.5; b < 1; b = b + 0.05) {
    //     double error = 0;
    //     for (int start = 0; start < startingPoints.size(); start++) {
    //         vector<double> point = startingPoints[start];
    //         vector<double> result = SimulatedAnnealing::OptimumFinder(function, point, maximum, maxIterations, bestT, b, bestSigma);
    //         error += function.getValue(result);
    //         cout << "T = " << bestT << " alpha = " << b << " sigma = " << bestSigma << " error = " << error << endl;
            
    //     }
    //     if (error < bestError) {
    //         bestAlpha = b;
    //         bestError = error;
    //     }
    //     cout << "----------------------------------------" << endl;
    //     cout << " T = " << bestT << " alpha = " << bestAlpha << " sigma = " << bestSigma << " error = " << bestError << endl;
    //     cout << "----------------------------------------" << endl;
    // }
    

   
    // for (double a = 100; a <= 1000; a += 100) {
    //     double error = 0;
    //     for (int start = 0; start < startingPoints.size(); start++) {
    //         vector<double> point = startingPoints[start];
    //         vector<double> result = SimulatedAnnealing::OptimumFinder(function, point, maximum, maxIterations, a, bestAlpha, bestSigma);
    //         double pom = function.getValue(result);
    //         error += pom;
    //         cout << "T = " << a << " alpha = " << bestAlpha << " sigma = " << bestSigma << " error = " << pom << endl;
            
    //     }
    //     if (error < bestError) {
    //         bestT = a;
    //         bestError = error;
    //     }
    //     cout << "----------------------------------------" << endl;
    //     cout << " T = " << bestT << " alpha = " << bestAlpha << " sigma = " << bestSigma << " error = " << bestError << endl;
    //     cout << "----------------------------------------" << endl;
    // }



    cout << " T = " << bestT << " alpha = " << bestAlpha << " sigma = " << bestSigma << " error = " << bestError << endl;
    for (int i = 0; i < startingPoints.size(); i++) {
        startingPoints[i][4] = fmod(startingPoints[i][4], acos(-1));
        vector<double> result = SimulatedAnnealing::OptimumFinder(function, startingPoints[i], maximum, maxIterations, bestT, bestAlpha, bestSigma);
        cout << "Best point:" << endl;
        for (int j = 0; j < result.size(); j++) {
            if (j == 4) {
                cout << fmod(result[4], acos(-1)) << " ";
                continue;
            }
            cout << result[j] << " ";
        }
        cout << endl;
        cout << "Value : " << function.getValue(result) << endl;
    }
    return 0;
}