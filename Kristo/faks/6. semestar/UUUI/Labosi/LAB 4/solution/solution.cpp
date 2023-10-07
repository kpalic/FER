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

map<vector<double>, double> trainData;
map<vector<double>, double> testData;
vector<int> architecture;

double sigmoid(double x) {
    return (1.0 / (1.0 + exp(-x)));
}

class Neuron {

    public:
    double bias;
    std::vector<double> weights;

    Neuron() {
        this->bias = 0;
    }
    Neuron(int numberOfInputs) {
        // inicijalizacija težina i biasa
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> normal_dist(0, 0.01);
        for (int i = 0; i < numberOfInputs; ++i) {
            // nasumično postavljanje težina u opsegu -1 do 1
            weights.push_back(normal_dist(gen));
        }
        bias = ((double) rand() / (RAND_MAX)) * 2 - 1;
    }

    double computeOutput(vector<double> input, bool lastLayer) {
        double netInput = inner_product(weights.begin(), weights.end(), input.begin(), 0.0) + bias;
        // Ako je ovaj neuron u posljednjem sloju, ne primjenjujemo aktivacijsku funkciju
        return lastLayer ? sigmoid(netInput) : netInput;
    }

    Neuron(Neuron parentOne, Neuron parentTwo) {
        vector<double> weights;
        for (int i = 0; i < parentOne.weights.size(); i++) {
            weights.push_back((parentOne.weights.at(i) + parentTwo.weights.at(i)) / 2.0);
        }
        this->bias = (parentOne.bias + parentTwo.bias) / 2.0;
        this->weights = weights;
    }
};

class Layer {

    public:
    std::vector<Neuron> neurons;
    bool applicastionLayer;

    Layer() {
        this->neurons = {};
    }

    Layer(int numberOfNeurons, bool applicationLayer, int numberOfInputs) {
        vector<Neuron> neurons;
        for (int i = 0; i < numberOfNeurons; i++) {
            neurons.push_back(Neuron(numberOfInputs));
        }
        this->applicastionLayer = applicationLayer;
        this->neurons = neurons;
    }

    Layer(Layer parentOne, Layer parentTwo) {
        vector<Neuron> neurons;
        for (int i = 0; i < parentOne.neurons.size(); i++) {
            neurons.push_back(Neuron(parentOne.neurons.at(i), parentTwo.neurons.at(i)));
        }
        this->applicastionLayer = parentOne.applicastionLayer;
        this->neurons = neurons;
    }


    vector<double> computeOutputs(vector<double> inputs) {
        vector<double> result;
        for (int i = 0; i < this->neurons.size(); i++) {
            result.push_back(this->neurons.at(i).computeOutput(inputs, this->applicastionLayer));
        }
        return result;
    }
};

class NeuralNet {

    public:
    vector<Layer> layers;

    NeuralNet() {
        this->layers = {};
    }

    NeuralNet(vector<int> numberOfNeuronsInLayer, int numberOfInputs) {
        vector<Layer> layers;
        bool applicationLayer = true;
        for (int i = 0; i <= numberOfNeuronsInLayer.size(); i++) {
            if (i == numberOfNeuronsInLayer.size()) {
                applicationLayer = false;
            } 
            if (i == 0) {
                layers.push_back(Layer(numberOfNeuronsInLayer.at(i), applicationLayer, numberOfInputs));
            }
            else if (i != numberOfNeuronsInLayer.size()) {
                layers.push_back(Layer(numberOfNeuronsInLayer.at(i), applicationLayer, numberOfNeuronsInLayer.at(i - 1)));
            }
            else {
                layers.push_back(Layer(1, applicationLayer, numberOfNeuronsInLayer.at(i - 1)));
            }
        }
        this->layers = layers;
    }

    NeuralNet(NeuralNet parentOne, NeuralNet parentTwo) {
        vector<Layer> layers;
        for (int i = 0; i < parentOne.layers.size(); i++) {
            layers.push_back(Layer(parentOne.layers.at(i), parentTwo.layers.at(i)));
        }
        this->layers = layers;
    }


    double computeOutput(vector<double> inputs) {
        vector<double> innerResult;
        for (int i = 0; i < this->layers.size(); i++) {
            if (i == 0) {
                innerResult = layers.at(i).computeOutputs(inputs);
            }
            else {
                innerResult = layers.at(i).computeOutputs(innerResult);
            }
        }
        double result = 0;
        for (auto j : innerResult) {
            result += j;
        }
        return result;
    }
};

vector<int> parseArchitecture(string nn) {
    vector<int> numbersOfNeuronsInLayer;
    stringstream ss(nn);
    string numberOfNeurons;
    while (getline(ss, numberOfNeurons, 's')) {
        numbersOfNeuronsInLayer.push_back(stoi(numberOfNeurons));
    }
    return numbersOfNeuronsInLayer;
}

double meanError(double x, double x0) {
    double result = pow((x - x0), 2);
    return result;
}

vector<pair<NeuralNet, double>> evaluate(vector<NeuralNet> population, map<vector<double>,double> data) {
    vector<pair<NeuralNet,double>> result;
    for (auto i : population) {
        double sumErrors;
        for (auto j = data.begin(); j != data.end(); j++) {
            double computedOutput = i.computeOutput(j->first);
            double realOutput = j->second;
            sumErrors += meanError(computedOutput, realOutput);
        }
        sumErrors = sumErrors / data.size();
        result.push_back(make_pair(i, sumErrors));
    }
    return result;
}

vector<NeuralNet> elitism(int bestOfPopulation, vector<pair<NeuralNet, double>> evaluation) {
    vector<NeuralNet> result;
    multimap<double, NeuralNet> ranked;
    for (auto i : evaluation) {
        ranked.insert(make_pair(i.second, i.first));

    }

    while (result.size() != bestOfPopulation) {
        auto it = ranked.begin();
        result.push_back(it->second);
        ranked.erase(it);
    }
    return result;
}

pair<NeuralNet, NeuralNet> selection (vector<pair<NeuralNet, double>> evaluation) {
    double sumErrors = 0;
    for (auto i : evaluation) {
        sumErrors += i.second;
    }

    vector<double> probabilities;
    for (auto i : evaluation) {
        probabilities.push_back((1 - (i.second / sumErrors)) / (evaluation.size() - 1));
    }

    random_device rd;
    mt19937 gen(rd());

    discrete_distribution<int> dist(probabilities.begin(), probabilities.end());

    // dva razlicita roditelja
    int parentOne = dist(gen);
    int parentTwo;
    do {
        parentTwo = dist(gen);
    } while (parentOne == parentTwo);

    return make_pair(evaluation.at(parentOne).first, evaluation.at(parentTwo).first);
}

NeuralNet mutate(NeuralNet child, double mutationProbability, double stddev) {

    random_device rd;
    mt19937 gen(rd());
    vector<double> probabilities;
    probabilities.push_back(mutationProbability);
    probabilities.push_back(1-mutationProbability);
    // cout << probabilities.at(0) << " " << probabilities.at(1) << " " << mutationProbability << endl;

    discrete_distribution<int> dist(probabilities.begin(), probabilities.end());
    
    NeuralNet result = NeuralNet();
    for (int i = 0; i < child.layers.size(); i++) { // za svaki layer
        Layer layer = Layer();
        for (int j = 0; j < child.layers.at(i).neurons.size(); j++) { // za svaki neuron u layeru
            Neuron neuron = Neuron();
            neuron.bias = child.layers.at(i).neurons.at(j).bias;
            bool mutation = dist(gen);
            // cout << mutation << endl;
            // cout << neuron.bias << ", " << mutation << ", " << probabilities.at(0) << ", " << probabilities.at(1) << endl;
            if (mutation) {                        // mutate bias
                normal_distribution<double> normal_dist(0, stddev);
                double noise = normal_dist(gen);
                neuron.bias += noise;
            }
            for (int k = 0; k < child.layers.at(i).neurons.at(j).weights.size(); k++) { // za svaku tezinu u neuronu
                bool mutation = dist(gen);
                neuron.weights.push_back(child.layers.at(i).neurons.at(j).weights.at(k));
                if (mutation) {
                    normal_distribution<double> normal_dist(0, stddev);
                    double noise = normal_dist(gen);
                    neuron.weights.at(k) += noise;                    
                }
            }
            layer.neurons.push_back(neuron);
        }
        layer.applicastionLayer = child.layers.at(i).applicastionLayer;
        result.layers.push_back(layer);
    }
    return result;
}

void printNeuralNet (NeuralNet myNet) {
    int layer = 1;
    for (auto i : myNet.layers) {
        cout << (i.applicastionLayer ? "" : "LAST ") << "Layer : " << layer << endl;
        layer++;
        int neuron = 1;
        for (auto j : i.neurons) {
            cout << "   Neuron: " << neuron << endl;
            neuron++;
            cout << "       bias: " << j.bias << endl;
            for (auto k : j.weights) {
                cout << "       weight: " << k << endl;
            }
        }
    }
}

vector<NeuralNet> GeneticAlgorithm(vector<int> architecture, 
                                   int numberOfInputs, 
                                   int popsize, 
                                   int numberOfIteration,
                                   map<vector<double>,double> data,
                                   int numberOfElite,
                                   double mutationProbability,
                                   double stddev) {

    vector<NeuralNet> population;
    for (int i = 0; i < popsize; i++) {
        population.push_back(NeuralNet(architecture, numberOfInputs));
    }

    vector<pair<NeuralNet, double>> evaluation = evaluate(population, data);

    for (int i = 1; i <= numberOfIteration; i++) {
        if (i % 2000 == 0) {
            double minError = 99999999;
            for (auto a : evaluation) {
                if (a.second < minError) {
                    minError = a.second;
                }
            }
            cout << "[Train error @" << i << "]: " << minError << endl;
        }

        vector<NeuralNet> newPopulation;
        bool eliteNNAdded = false;
        while (newPopulation.size() != popsize) {
            if (!eliteNNAdded) {
                vector<NeuralNet> elite = elitism(numberOfElite, evaluation);
                for (auto j : elite) {
                    newPopulation.push_back(j);
                }
                eliteNNAdded = true;
            }
            pair<NeuralNet, NeuralNet> parents = selection(evaluation);
            NeuralNet child = NeuralNet(parents.first, parents.second);
            newPopulation.push_back(mutate(child, mutationProbability, stddev));
        }

        population = newPopulation;
        evaluation = evaluate(population, data);
    }
    return elitism(1, evaluate(population, data));
}


void parseData(string fileName, bool train) {
    ifstream file(fileName);
    bool header = true;
    if (file.is_open()) {
        string line;
        while (getline(file, line)) {
            if (header) {
                header = false;
                continue;
            }
            vector<double> key;
            double mapValue;
            string value;
            stringstream ss(line);
            while(getline(ss, value, ',')) {
                key.push_back(stod(value));
            }
            mapValue = key.back();
            key.pop_back();
            if (train) {
                trainData.insert(make_pair(key, mapValue));
            }
            else {
                testData.insert(make_pair(key, mapValue));
            }
        }
    }
}

int main(int argc, char* argv[]) {
    
    string trainFile = (string)argv[2];
    string testFile = (string)argv[4];
    parseData(trainFile, true);
    parseData(testFile, false);

    architecture = parseArchitecture((string)argv[6]);
    
    int popsize = stoi(argv[8]);
    int elite = stoi(argv[10]);
    double probabilityOfMutation = stod(argv[12]);
    double GaussianCoefficient = stod(argv[14]);
    int iter = stoi(argv[16]);

    vector<NeuralNet> bestNN = GeneticAlgorithm(architecture, 
                                                trainData.begin()->first.size(), 
                                                popsize, 
                                                iter, 
                                                trainData, 
                                                elite,
                                                probabilityOfMutation,
                                                GaussianCoefficient);


    vector<pair<NeuralNet, double>> testEvaluation = evaluate(bestNN, testData);
    cout << "[Test error]: " << (testEvaluation.at(0).second) << endl;
}