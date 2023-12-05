#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include <iostream>
#include <string>
#include <fstream>
#include <set>
#include <utility>
#include <sstream>

#include "Node.hpp"

using namespace std;

class Configuration {
    public:
        set<Operator> functions;
        pair<double, double> constantRange;
        int populationSize;
        int tournamentSize;
        double costEvaluationMax;
        double mutationProbability;
        int maxTreeDepth;
        bool useLinearScaling;

        void printConfiguration() {
            cout << "Possible functions: ";
            for (auto it = functions.begin(); it != functions.end(); it++) {
                if ((*it) == Operator::Plus) {
                    cout << "+ ";
                } else if ((*it) == Operator::Minus) {
                    cout << "- ";
                } else if ((*it) == Operator::Multiply) {
                    cout << "* ";
                } else if ((*it) == Operator::Divide) {
                    cout << "/ ";
                } else if ((*it) == Operator::Sin) {
                    cout << "sin ";
                } else if ((*it) == Operator::Cos) {
                    cout << "cos ";
                } else if ((*it) == Operator::Sqrt) {
                    cout << "sqrt ";
                } else if ((*it) == Operator::Log) {
                    cout << "log ";
                } else if ((*it) == Operator::Exp) {
                    cout << "exp ";
                }
            }
            cout << endl;
            cout << "Constant range: " << constantRange.first << " to " << constantRange.second << endl;
            cout << "Population size: " << populationSize << endl;
            cout << "Tournament size: " << tournamentSize << endl;
            cout << "Cost evaluation max: " << costEvaluationMax << endl;
            cout << "Mutation probability: " << mutationProbability << endl;
            cout << "Max tree depth: " << maxTreeDepth << endl;
            cout << "Use linear scaling: " << useLinearScaling << endl;
        }

        Configuration(string filename) {
            ifstream file;
            file.open(filename);
            if (!file.is_open()) {
                cout << "File not found" << endl;
                throw;
            }

            string line;
            getline(file, line); // possible functions
            line = line.substr(line.find(':') + 1);
            vector<string> tokens;
            stringstream ss(line);
            string token;
            while (getline(ss, token, ',')) {
                tokens.push_back(token.substr(1, token.length()));
            }

            for (int i = 0; i < tokens.size(); i++) {
                if (tokens[i] == "+") {
                    functions.insert(Operator::Plus);
                } else if (tokens[i] == "-") {
                    functions.insert(Operator::Minus);
                } else if (tokens[i] == "*") {
                    functions.insert(Operator::Multiply);
                } else if (tokens[i] == "/") {
                    functions.insert(Operator::Divide);
                } else if (tokens[i] == "sin") {
                    functions.insert(Operator::Sin);
                } else if (tokens[i] == "cos") {
                    functions.insert(Operator::Cos);
                } else if (tokens[i] == "sqrt") {
                    functions.insert(Operator::Sqrt);
                } else if (tokens[i] == "log") {
                    functions.insert(Operator::Log);
                } else if (tokens[i] == "exp") {
                    functions.insert(Operator::Exp);
                }
            }

            getline(file, line); // constant range
            line = line.substr(line.find(':') + 1);
            constantRange.first = stod(line.substr(0, line.find(',')));
            
            constantRange.second = stod(line.substr(line.find(',') + 1));

            getline(file, line); // population size
            line = line.substr(line.find(':') + 1);
            populationSize = stoi(line);

            getline(file, line); // tournament size
            line = line.substr(line.find(':') + 1);
            tournamentSize = stoi(line);

            getline(file, line); // cost evaluation max
            line = line.substr(line.find(':') + 1);
            costEvaluationMax = stod(line);

            getline(file, line); // mutation probability
            line = line.substr(line.find(':') + 1);
            mutationProbability = stod(line);

            getline(file, line); // max tree depth
            line = line.substr(line.find(':') + 1);
            maxTreeDepth = stoi(line);

            getline(file, line); // use linear scaling
            line = line.substr(line.find(':') + 1);
            useLinearScaling = stoi(line);


            // printConfiguration();
        }
};


#endif