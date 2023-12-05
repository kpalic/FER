#ifndef F3_HPP
#define F3_HPP

#include "IFunction.hpp"

using namespace std;

class F3 : public IFunction {
    private:
        int n;

    public:
        F3 () {
            this->n = 2;
        }
        F3(int n) {
            this->n = n;
        }

        double getValue(vector<double> X) override {
            double sum = 0;
            for (int i = 1; i <= X.size(); i++) {
                sum += pow(X[i - 1] - i, 2);
            }
            return sum;
        }

        int getNumberOfParameters() override {
            return n;
        }
};


#endif // F3