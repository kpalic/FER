#ifndef F5_HPP
#define F5_HPP

#include "IFunction.hpp"

using namespace std;

class F5 : public IFunction {
        private:
            int n;

        public:

            F5(int n) {
                this->n = n;
            }

            double getValue(vector<double> X) override {
                double sumXi = 0;
                for (int i = 0; i < X.size(); i++) {
                    sumXi += pow(X[i], 2);
                }
                return 0.5 + ((pow(sin(sqrt(sumXi)), 2) - 0.5) / pow(1 + 0.001 * sumXi, 2));
            }
    
            int getNumberOfParameters() override {
                return n;
            }
};

#endif // F5