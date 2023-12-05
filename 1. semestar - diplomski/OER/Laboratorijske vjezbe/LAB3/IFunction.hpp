#ifndef IFUNCTION_HPP
#define IFUNCTION_HPP

#include <vector>

class IFunction{

    public:
        virtual ~IFunction () = default;

        virtual int getNumberOfVariables() = 0;

        virtual double getValue(std::vector<double> point) = 0;

        virtual std::vector<double> getGradient(std::vector<double> point) = 0;

};

#endif