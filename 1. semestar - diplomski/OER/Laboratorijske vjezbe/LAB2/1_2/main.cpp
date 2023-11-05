#include <iostream>
#include <random>
#include <vector>

#include "IFunction.hpp"
#include "F1.hpp"
#include "F2.hpp"
#include "NumOptAlgorithms.hpp"


using namespace std;

int main (int argc, char* argv[]) {

    if (argc < 3) {
        cout << "Neispravan broj argumenata" << endl;
        return 1;
    }

    int option = stoi(argv[1]);
    int maxIter = stoi(argv[2]);

    F1 function1;
    F2 function2;
    vector<double> startingPoint;
    if (argc > 3) {
        startingPoint.push_back(stod(argv[3]));
        startingPoint.push_back(stod(argv[4]));
    }
    else {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-1000, 1000);

        startingPoint.push_back(dis(gen));
        startingPoint.push_back(dis(gen));
    }

    if (option == 1) {
        vector<double> result = NumOptAlgorithms::gradientDescent(function1, maxIter, startingPoint);
        cout << "Konacni rezultat: [" << result[0] << ", " << result[1] << "]" << endl;
    } 
    else if (option == 2) {
        vector<double> result = NumOptAlgorithms::gradientDescent(function2, maxIter, startingPoint);
        cout << "Konacni rezultat: [" << result[0] << ", " << result[1] << "]" << endl;

    }
    else {
        std::cerr << "Nepoznata opcija." << std::endl;
    }

}