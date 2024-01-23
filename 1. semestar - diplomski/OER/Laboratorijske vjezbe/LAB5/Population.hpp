#ifndef POPULATION_HPP
#define POPULATION_HPP

#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>

#include "BinaryTree.hpp"
#include "Configuration.hpp"
#include "Function.hpp"

using namespace std;

class Population {
    private:
        vector<BinaryTree> trees;
        Function* function;
        Configuration* config;
        int generation;

    public:
        Population(Function* function, Configuration* config) {
            this->function = function;
            this->config = config;
            this->generation = 0;
        }

        vector<BinaryTree> getTrees() {
            return this->trees;
        }
        void setTrees(vector<BinaryTree> trees) {
            this->trees = trees;
        }

        Function* getFunction() {
            return this->function;
        }
        void setFunction(Function* function) {
            this->function = function;
        }

        Configuration* getConfig() {
            return this->config;
        }
        void setConfig(Configuration* config) {
            this->config = config;
        }

        int getGeneration() {
            return this->generation;
        }
        void setGeneration(int generation) {
            this->generation = generation;
        }

        void printPopulation() {
            cout << "Generation: " << this->generation << endl;
            for (int i = 0; i < this->trees.size(); i++) {
                cout << "Tree: " << i << endl;
                this->trees[i].printTree();
            }
        }

        void generatePopulation() {
            int populationSize = this->config->getPopulationSize();
            int currentPopulationSize;
            int maxTreeDepth = this->config->getMaxTreeDepth();

            double percentagePerDepth = 1.0 / (maxTreeDepth - 1);
            int numFullTrees = floor(populationSize * percentagePerDepth / 2);
            int numGrowTrees = numFullTrees;

            // cout << "Population size: " << populationSize << endl;
            // cout << "Percentage per depth: " << percentagePerDepth << endl;

            // cout << "Generating " << populationSize / 2 << " full trees" << endl;
            // cout << "Generating " << populationSize / 2 << " grow trees" << endl;

            // for (int i = 2; i <= maxTreeDepth; i++) {
            //     cout << "Generating depth " << i << endl;
            //     cout << "Generating " << numFullTrees << " full trees" << endl;
            //     cout << "Generating " << numGrowTrees << " grow trees" << endl;
            // }

            BinaryTree* tree = BinaryTree::generateTree(buildTreeMethod::FULL, 5, this->config->getConstantRange(), this->config->getOperators(), this->function->getVariableNames());
            tree->printTree();


        }
        
};

#endif