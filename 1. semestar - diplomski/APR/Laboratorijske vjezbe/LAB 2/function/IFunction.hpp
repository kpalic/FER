#ifndef IFUNCTION_HPP
#define IFUNCTION_HPP

#include <vector>
#include <cmath>
#include <utility>

using namespace std;

class IFunction {
    public:
    virtual double getValue(vector<double> X) = 0;
    virtual int getNumberOfParameters() = 0;

    static pair<double, double> unimodalInterval(vector<double> startingPoint, 
                                                 int variableIndex, 
                                                 double h, 
                                                 IFunction& f) {

        vector<double> m = startingPoint;
        vector<double> l = m;
        l[variableIndex] -= h;
        vector<double> r = m;
        r[variableIndex] += h;

        double fl = f.getValue(l);
        double fm = f.getValue(m);
        double fr = f.getValue(r);
        
        int step = 1;

        if(fm < fr && fm < fl) 
        {
            return (make_pair(l[variableIndex], r[variableIndex]));
        }
        else if(fm > fr) 
        {
            do {
                l = m;
                m = r;
                fm = fr;
                step *= 2;
                r[variableIndex] = m[variableIndex] + (h * step);
                fr = f.getValue(r);
            } while(fm > fr);
        }
        else 
        {
            do {
                r = m;
                m = l;
                fm = fl;
                step *= 2;
                l[variableIndex] = m[variableIndex] - (h * step);
                fl = f.getValue(l);
            } while(fm > fl);
        }

        return make_pair(l[variableIndex], r[variableIndex]);
    }
};

#endif // IFUNCTION_HPP