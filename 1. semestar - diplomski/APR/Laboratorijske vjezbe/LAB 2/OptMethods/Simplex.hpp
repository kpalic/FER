#ifndef SIMPLEX_HPP
#define SIMPLEX_HPP

#include "../function/IFunction.hpp"
#include <iostream>
#include <algorithm>
#include <functional>
#include <map>

using namespace std;

class Simplex {
    private:
        vector<double> bounds;
        bool intervalGiven = false;

        double alpha;
        double beta;
        double gama;

    public:
    int iteration = 0;

        Simplex(vector<double> bounds) {
            this->bounds = bounds;
            this->intervalGiven = true;
        }
        Simplex(vector<double> bounds, double alpha, double beta, double gama) {
            this->bounds = bounds;
            this->intervalGiven = true;
            this->alpha = alpha;
            this->beta = beta;
            this->gama = gama;
        }
        Simplex(double alpha, double beta, double gama) {
            this->intervalGiven = false;
            this->alpha = alpha;
            this->beta = beta;
            this->gama = gama;
        }
        
        vector<double> reflect(vector<double> point, vector<double> centroid, double factor) {
            vector<double> result(point.size());
            for (int i = 0; i < point.size(); ++i) {
                result[i] = centroid[i] + factor * (centroid[i] - point[i]);
            }
            return result;
        }


        // Ulazne velicine: X0, alpha, beta, gama, epsilon        
        vector<double> findOptimum(IFunction& f,
                                   vector<double> startingPoint,
                                   double delta,
                                   double epsilon, 
                                   int maxIterations)  
        {
            // izracunaj tocke simpleksa X[i], i = 0..n;
            int n = startingPoint.size();
            vector<vector<double>> simplex(n+1, startingPoint);   
            for (int i = 0; i < n; i++) {
                simplex[i][i] += delta;
            }

            // for (int i = 0; i < simplex.size(); i++) {
            //     for (int j = 0; j < simplex[i].size(); j++) {
            //         cout << simplex[i][j] << " ";
            //     }
            //     cout << " ---> " << f.getValue(simplex[i]) << endl;
            // }
            
            this->iteration = 0;
            while (iteration < maxIterations) {
                sort(simplex.begin(), simplex.end(), [&](const vector<double>& a, const vector<double>& b) {
                    return (f.getValue(a) < f.getValue(b));
                });

                // for (int i = 0; i < simplex.size(); i++) {
                //     for (int j = 0; j < simplex[i].size(); j++) {
                //         cout << simplex[i][j] << " ";
                //     }
                //     cout << " ---> " << f.getValue(simplex[i]) << endl;
                // }

                vector<double> centroid(n, 0);
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        centroid[j] += simplex[i][j];
                    }
                }

                for (double& value : centroid) {
                    value /= n;
                }

                vector<double> reflected = reflect(simplex[n], centroid, alpha);
                double reflectedValue = f.getValue(reflected);

                if (reflectedValue < f.getValue(simplex[0])) {
                    vector<double> expanded = reflect(simplex[n], centroid, gama);
                    simplex[n] = (f.getValue(expanded) < reflectedValue) ? expanded : reflected;
                }
                else if (reflectedValue < f.getValue(simplex[n-1])) {
                    simplex[n] = reflected;
                }
                else {
                    vector<double> contracted = reflect(simplex[n], centroid, beta);
                    if (f.getValue(contracted) < f.getValue(simplex[n])) {
                        simplex[n] = contracted;
                    }
                    else {
                        for (int i = 1; i <= n; i++) {
                            for (int j = 0; j < n; j++) {
                                simplex[i][j] = simplex[0][j] + beta * (simplex[i][j] - simplex[0][j]);
                            }
                        }
                    }
                }
                double size = 0;
                for (int i = 0; i <= n; ++i) {
                    size += pow(f.getValue(simplex[i]) - f.getValue(simplex[0]), 2);
                }
                size = sqrt(size / (n + 1));
                if (size < epsilon) {
                    break;
                }
            iteration++;
            // for (int i = 0; i < simplex.size(); i++) {
            //     for (int j = 0; j < simplex[i].size(); j++) {
            //         cout << simplex[i][j] << " ";
            //     }
            //     cout << endl;
            // }
            // cout << "-----------------------------------------" << endl;
            }
            // cout << "Number of iterations: " << iteration << endl;
            return simplex[0];
        }        
};

#endif // SIMPLEX_HPP