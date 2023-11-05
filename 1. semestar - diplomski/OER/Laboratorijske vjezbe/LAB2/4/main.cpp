#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <string>
#include <filesystem>

#include "F4.hpp"
#include "IFunction.hpp"
#include "NumOptAlgorithms.hpp"

using namespace std;    

int main (int argc, char* argv[]) {
    if (argc < 3) {
        cout << "Neispravan broj argumenata" << endl;
        return 1;
    }

    int maxIter = stoi(argv[1]);
    string filename = argv[2];    
    Matrix A(20, 5);
    Matrix b(20, 1);

    string base = filesystem::current_path().string();
    string fullPath = base + "\\" + filename;

    ifstream file(fullPath);
    cout << base << "\\" << filename << endl;
    if (!file.is_open()) {
        cout << "Neispravan naziv datoteke" << endl;
        return 1;
    }
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

    vector<double> startingPoint(6);
    F4 function(A, b);

    vector<double> result = NumOptAlgorithms::gradientDescent(function, maxIter, startingPoint);
    Matrix x = Matrix(result.size(), 1);
    cout << "Konacni rezultat: " << endl;
    for (int i = 0; i < result.size(); i++) {
        cout << result[i] << " ";
        x.values[i][0] = result[i];
    }
    cout << endl;
    cout << "Vrijednost funkcije: " << function.getValue(result) << endl;
    return 0;
}