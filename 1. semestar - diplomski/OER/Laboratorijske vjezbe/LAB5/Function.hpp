#ifndef FUNCTION_HPP
#define FUNCTION_HPP

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <map>
#include <sstream>

using namespace std;

class Function {
    private:
        vector<string> variableNames;
        vector<vector<double>> input;
        vector<double> output;

    public:
        Function(string filename) {

            ifstream file;
            file.open(filename);
            if (!file) {
                cout << "Unable to open file" << endl;
                exit(1);
            }
            string line;
            getline(file, line);
            vector<string> tokens;
            string token;
            stringstream ss(line);
            while (getline(ss, token, '\t')) {
                tokens.push_back(token);
            }
            int numVariables = tokens.size() - 1;
            vector<vector<double>> input;
            vector<double> output;
            vector<double> firstRow;
            for (int i = 0; i < numVariables; i++) {
                string name = "X" + to_string(i);
                variableNames.push_back(name);
                firstRow.push_back(stod(tokens[i]));
            }
            input.push_back(firstRow);
            output.push_back(stod(tokens[numVariables]));

            while (getline(file, line)) {
                vector<string> tokens;
                string token;
                stringstream ss(line);
                while (getline(ss, token, '\t')) {
                    tokens.push_back(token);
                }
                vector<double> row;
                for (int i = 0; i < numVariables; i++) {
                    row.push_back(stod(tokens[i]));
                }
                input.push_back(row);
                output.push_back(stod(tokens[numVariables]));
            }
            this->variableNames = variableNames;
            this->input = input;
            this->output = output;

            // printFunction();

        }

        vector<string> getVariableNames() {
            return this->variableNames;
        }
        vector<vector<double>> getInput() {
            return this->input;
        }
        vector<double> getOutput() {
            return this->output;
        }

        void printFunction() {
            cout << "Function: " << endl;
            for (int i = 0; i < variableNames.size(); i++) {
                cout << variableNames[i] << "\t";
            }
            cout << endl;
            for (int i = 0; i < input.size(); i++) {
                for (int j = 0; j < input[i].size(); j++) {
                    cout << input[i][j] << "\t";
                }
                cout << " -> " << output[i] << endl;
            }
        }

        
};
#endif