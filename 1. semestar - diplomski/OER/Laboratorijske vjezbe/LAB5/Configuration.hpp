#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include <iostream>
#include <string>
#include <fstream>
#include <set>
#include <vector>
#include <sstream>

using namespace std;

class Configuration {
    private:
        set<string> operators;
        pair<double, double> constantRange;
        int populationSize;
        int tournamentSize;
        double maxError;
        double mutationProbability;
        int maxTreeDepth;
        bool useLinearScaling;

    public:
        void printConfiguration() {
            cout << "Possible functions: ";
            for (auto it = operators.begin(); it != operators.end(); it++) {
                cout << *it << " ";
            }
            cout << endl;
            cout << "Constant range: " << constantRange.first << " to " << constantRange.second << endl;
            cout << "Population size: " << populationSize << endl;
            cout << "Tournament size: " << tournamentSize << endl;
            cout << "Cost evaluation max: " << maxError << endl;
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
                operators.insert(tokens[i]);
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
            maxError = stod(line);

            getline(file, line); // mutation probability
            line = line.substr(line.find(':') + 1);
            mutationProbability = stod(line);

            getline(file, line); // max tree depth
            line = line.substr(line.find(':') + 1);
            maxTreeDepth = stoi(line);

            getline(file, line); // use linear scaling
            line = line.substr(line.find(':') + 1);
            useLinearScaling = stoi(line);

            file.close();
        }

        set<string> getOperators() {
            return this->operators;
        }
        void setOperators(set<string> operators) {
            this->operators = operators;
        }

        pair<double, double> getConstantRange() {
            return this->constantRange;
        }
        void setConstantRange(pair<double, double> constantRange) {
            this->constantRange = constantRange;
        }

        int getPopulationSize() {
            return this->populationSize;
        }
        void setPopulationSize(int populationSize) {
            this->populationSize = populationSize;
        }

        int getTournamentSize() {
            return this->tournamentSize;
        }
        void setTournamentSize(int tournamentSize) {
            this->tournamentSize = tournamentSize;
        }

        double getMaxError() {
            return this->maxError;
        }
        void setMaxError(double maxError) {
            this->maxError = maxError;
        }

        double getMutationProbability() {
            return this->mutationProbability;
        }
        void setMutationProbability(double mutationProbability) {
            this->mutationProbability = mutationProbability;
        }

        int getMaxTreeDepth() {
            return this->maxTreeDepth;
        }
        void setMaxTreeDepth(int maxTreeDepth) {
            this->maxTreeDepth = maxTreeDepth;
        }

        bool getUseLinearScaling() {
            return this->useLinearScaling;
        }
        void setUseLinearScaling(bool useLinearScaling) {
            this->useLinearScaling = useLinearScaling;
        }

};

#endif