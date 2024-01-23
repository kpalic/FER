#ifndef ALGORITHM_HPP
#define ALGORITHM_HPP

#include <vector>
#include <cmath>

using namespace std;

class Algorithm {
    public:

    vector<double> normalizeGradient(vector<double> gradient) {
        vector<double> normalizedGradient(gradient.size());
        double norm = getNorm(gradient);
        for (int i = 0; i < gradient.size(); i++) {
            normalizedGradient[i] = gradient[i] / norm;
        }
        // cout << "Gradient: " << endl;
        // for (int i = 0; i < gradient.size(); i++) {
        //     cout << gradient[i] << " ";
        // }
        // cout << endl;

        // cout << "Normalized gradient: " << endl;
        // for (int i = 0; i < normalizedGradient.size(); i++) {
        //     cout << normalizedGradient[i] << " ";
        // }
        // cout << endl;

        return normalizedGradient;
    }

    double getNorm(vector<double> gradient) {
        double norm = 0;
        for (int i = 0; i < gradient.size(); i++) {
            norm += pow(gradient[i], 2);
        }
        return sqrt(norm);
    }
    
};

#endif