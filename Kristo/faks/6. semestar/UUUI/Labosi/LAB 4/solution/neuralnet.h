#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>
#include <cmath>
#include <numeric>
#include <map>
#include <string>
#include <sstream>
#include <iostream>
#include <random>
#include <fstream>
#include <set>

using namespace std;

class Neuron {

    public:
        double bias;
        std::vector<double> weights;
        
        // Konstruktor, metode...
        Neuron();
        Neuron(int numberOfInputs);
        Neuron(Neuron parentOne, Neuron parentTwo);
        double computeOutput(vector<double> input, bool lastLayer);    
};

class Layer {

    public:
        std::vector<Neuron> neurons;
        bool applicastionLayer;

        // Konstruktor, metode...
        Layer();
        Layer(int numberOfNeurons, bool applicationLayer, int numberOfInputs);
        Layer(Layer parentOne, Layer parentTwo);
        vector<double> computeOutputs(vector<double> inputs);
};

class NeuralNet {

    public:
        std::vector<Layer> layers;

        // Konstruktor, metode...
        NeuralNet();
        NeuralNet(vector<int> numberOfNeuronsInLayer, int numberOfInputs);
        NeuralNet(NeuralNet parentOne, NeuralNet parentTwo);
        double computeOutput(vector<double> inputs);  
};

double sigmoid(double x);
vector<int> parseArchitecture(string nn);
double meanError(double x, double x0);
vector<pair<NeuralNet, double>> evaluate(vector<NeuralNet> population, map<vector<double>,double> data);
vector<NeuralNet> elitism(int bestOfPopulation, vector<pair<NeuralNet, double>> evaluation);
pair<NeuralNet, NeuralNet> selection (vector<pair<NeuralNet, double>>);
NeuralNet mutate(NeuralNet child, double mutationProbability, double stddev);
vector<NeuralNet> GeneticAlgorithm(vector<int> architecture, 
                                   int numberOfInputs, 
                                   int popsize, 
                                   int numberOfIteration,
                                   map<vector<double>,double> data,
                                   int numberOfElite,
                                   double mutationProbability,
                                   double stddev);
                                   
#endif // NEURALNET_H
