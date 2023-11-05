#include <iostream>
#include <vector>
#include "IFunction.hpp"

using namespace std;

class F1 : public IFunction {
    private:
        double minimum = 0;
    public:

        double getMinimum() override {
            return this->minimum;
        }

        int getNumberOfVariables() override {
            return 2;
        }

        double getValue(vector<double> point) override {
            if (point.size() != 2) {
                throw invalid_argument("Invalid dimension for function F1.");
            }
            double x1 = point[0];
            double x2 = point[1];

            return (x1 * x1) + ((x2 - 1) * (x2 - 1));
        }

        vector<double> getGradient(vector<double> point) override {
            if (point.size() != 2) {
                throw invalid_argument("Invalid dimension for function F1.");
            }
            double x1 = point[0];
            double x2 = point[1];
            
            // f(x1, x2) = x1^2 + (x2−1)^2;
            double g1 = 2*x1;
            double g2 = 2*(x2 - 1);

            vector<double> gradient;
            gradient.push_back(g1);
            gradient.push_back(g2);

            return gradient;
        }
};