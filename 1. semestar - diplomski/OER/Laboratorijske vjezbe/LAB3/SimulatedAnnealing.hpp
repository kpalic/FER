
#include <iostream>
#include <vector>
#include <random>
#include "F4.hpp"

#define EPSILON 10e-20

using namespace std;

class SimulatedAnnealing {
    public:

    static vector<double> OptimumFinder(F4 function,
                                 vector<double> startingPoint, 
                                 bool maximum, 
                                 int maxIterations,
                                 double startingTemperature,
                                 double alpha,
                                 double sigma) {
        
        random_device rd;
        mt19937 generator(rd());

        double temperature = startingTemperature;
        vector<double> currentPoint = startingPoint;
        int iteration = 1;
        double lastValue = maximum ? -10e50 : 10e50;
        double value = 0;
        double deltaE = 0;
        int neighborIndex = 0;
        bool accept = false;
        bool acceptBad = false;

        while (temperature > 0.01) {
            iteration = 1;
            while (iteration < maxIterations) {
                // cout << "Iteration " << iteration << endl;
                accept = false;
                value = function.getValue(currentPoint);
                vector<double> nextPoint = generateNeighbor(neighborIndex, currentPoint, sigma, generator);
                double nextValue = function.getValue(nextPoint);
                deltaE = nextValue - value;
                // cout << "Current value " << value << endl;
                // cout << "Next value " << nextValue << endl;

                // cout << "current point" << endl;
                // for (int i = 0; i < currentPoint.size(); i++) {
                //     cout << currentPoint[i] << " ";
                // }
                // cout << endl;
                // cout << "next point" << endl;
                // for (int i = 0; i < currentPoint.size(); i++) {
                //     cout << nextPoint[i] << " ";
                // }
                // cout << endl;
                // cout << "Iteration : " << iteration << " deltaE : " << nextValue - value << " T: " << temperature << endl;

                if ((maximum && deltaE > 0) || (!maximum && deltaE < 0)) {
                        lastValue = value;
                        value = nextValue;
                        currentPoint = nextPoint;
                }
                else {
                    acceptBad = acceptInferior(temperature, deltaE, generator);
                    if (acceptBad) {
                        lastValue = value;
                        value = nextValue;
                        currentPoint = nextPoint;
                        // temperature = alpha * temperature;
                    }
                }
                // cout << "\n\n" << endl;
                neighborIndex = (neighborIndex + 1) % currentPoint.size();
                iteration++;
            }
            temperature *= alpha;
        //     cout << "Optimal value" << endl;
        //     for (int i = 0; i < currentPoint.size(); i++) {
        //         cout << currentPoint[i] << " ";
            // }
        }
        // cout << endl;
        // cout << "Value : " << function.getValue(currentPoint) << " deltaE : " << value - lastValue << endl;
        return currentPoint;
    }

    static bool acceptInferior(double temperature, double deltaE, mt19937& generator) {
        double p = exp(-deltaE / temperature);
        bernoulli_distribution d(p);
        // cout << "p : " << p << endl;
        return d(generator);
    }

    static vector<double> generateNeighbor(int neighborIndex, const vector<double>& point, double sigma, mt19937& generator) {
        // Normalna distribucija s očekivanjem 0 i disperzijom sigma
        normal_distribution<double> distribution(0.0, sigma);
        
        // Novi vektor za susjedno rješenje
        vector<double> neighbor = point;
        neighbor[neighborIndex] = point[neighborIndex] + distribution(generator);

        return neighbor;
    }
};