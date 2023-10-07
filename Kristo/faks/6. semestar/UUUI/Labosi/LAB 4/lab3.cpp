#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <set>
#include <vector>
#include <algorithm>
#include <memory>
#include <map>
#include <stack>
#include <cmath>
#include <iomanip>
#include <cstring>

using namespace std;

class Node {
    public:
    string feature;
    string branchValue;
    vector<Node> childrenNodes;
    shared_ptr<Node> parentNode = nullptr;
    string maxFrequencyValue;

    bool leaf = false;
    string leafValue;

    Node(string feature, string branchValue) : feature(feature), branchValue(branchValue) {}
    Node(bool leaf, string branchValue, string leafValue) : leaf(leaf), branchValue(branchValue), leafValue(leafValue) {}
    Node(bool leaf) : leaf(leaf) {}
};

vector<map<string, string>> values;
vector<map<string, string>> trainValues;
vector<string> header;
string label;
map<string,map<string,int>> confusionMatrix;


map<string, int> maxFrequency(vector<map<string,string>> values) {
    map<string,int> frequency;
    
    for (auto i : values) {
        string label = i[header.back()];
        if (frequency.find(label) == frequency.end()) {
            frequency.insert(make_pair(label, 1));
        } 
        else {
            frequency[label] = frequency[label] + 1;
        }
    }
    int noMaxLabel = 0;
    string maxLabel;
    for (auto i = frequency.begin(); i != frequency.end(); i++) {
        if (i->second > noMaxLabel) {
            noMaxLabel = i->second;
            maxLabel = i->first;
        }
    }
    map<string, int> result;
    result.insert(make_pair(maxLabel, noMaxLabel));
    return result;
}

double entropy(vector<map<string,string>> values, vector<string> header) {
    map<string, int> frequency;
    int noOfValues = values.size();

    for (auto i : values) {
        string label = i[header.back()];
        if (frequency.find(label) == frequency.end()) {
            frequency.insert(make_pair(label, 1));
        } 
        else {
            frequency[label] = frequency[label] + 1;
        }
    }

    double entropyResult;
    for (auto i = frequency.begin(); i != frequency.end(); i++) {
        double pValue = i->second / (noOfValues * 1.0);
        entropyResult -= (pValue * log2(pValue));
    }

    return entropyResult;
}

double informationGain(vector<map<string,string>> values, 
                       vector<string> features, 
                       string feature) 
{    
    double infGain = entropy(values, features);

    map<string, int> frequency;
    for (auto i : values) {
        if (frequency.find(i[feature]) == frequency.end()) {
            frequency.insert(make_pair(i[feature], 1));
        }
        else {
            frequency[i[feature]] += 1;
        }
    }

    for (auto j = frequency.begin(); j != frequency.end(); j++) {
        vector<string> newFeatures = features;
        vector<map<string, string>> newValues;
        for (auto k = newFeatures.begin(); k != newFeatures.end(); k++) {
            if ((*k) == feature) {
                newFeatures.erase(k);
                break;
            }
        }

        for (auto l = values.begin(); l != values.end(); l++) {
            if (l->find(feature)->second == j->first) {
                map<string,string> newLine = *l;
                newLine.erase(newLine.find(feature));
                newValues.push_back(newLine);
            }
        }

        double ratio = j->second / (values.size() * 1.0);
        double newEntropy = entropy(newValues, newFeatures);
        infGain -= (ratio * newEntropy);
    }

    return infGain;
}

void parseInput(string input, bool headerInput = false) {
    
    stringstream ss(input);
    vector<string> row;
    string value;

    int i = 0;
    while(getline(ss, value, ',')) {
        row.push_back(value);
    }

    if (headerInput) {
        header = row;
        label = header.back();
    }
    else {
        map<string,string> inputValues;
        for (int i = 0; i != header.size(); i++) {
            inputValues.insert(make_pair(header.at(i), row.at(i)));
        }
        values.push_back(inputValues);
    }
}

bool headerTrain = false;
void parseTrainInput(string line) {
    stringstream ss(line);
    vector<string> row;
    string value;

    while(getline(ss, value, ',')) {
        row.push_back(value);
    }
    
    map<string,string> inputValues;
    for (int i = 0; i != header.size(); i++) {
        inputValues.insert(make_pair(header.at(i), row.at(i)));
    }
    
    if (!headerTrain) {
        headerTrain = true;
    }
    else {
        trainValues.push_back(inputValues);
    }
}

Node algorithmID3 (vector<map<string,string>> newValues,
                   vector<map<string,string>> oldValues,
                   vector<string> algoFeatures,
                   string branchValue,
                   shared_ptr<Node> parentNode,
                   int depth = 0,
                   int maxDepth = 999)
{
    if (newValues.size() == 0) {
        map<string, int> mostFrequent = maxFrequency(oldValues);
        Node leaf = Node(true, branchValue, mostFrequent.begin()->first);
        
        leaf.parentNode = parentNode;
        return leaf;
    }

    map<string,int> mostFrequent = maxFrequency(newValues);
    if (algoFeatures.size() == 1 || newValues.size() == mostFrequent.begin()->second || depth == maxDepth) {
        Node leaf = Node(true, branchValue, mostFrequent.begin()->first);
        leaf.parentNode = parentNode;
        return leaf;
    }

    double infGain = 0;
    string nextFeature;
    map<string, double> infGains;
    for (auto i : algoFeatures) {
        if (i != label) {
            infGains.insert(make_pair(i, informationGain(newValues, algoFeatures, i)));
        }
    } 
    for (auto i = infGains.begin(); i != infGains.end(); i++) {
        cout << "IG(" << i->first << ")=" << i->second << " "; 
        if (i->second > infGain) {
            infGain = i->second;
            nextFeature = i->first;
        }
    }
    cout << endl;

    set<string> nextFeatureValues;
    for (auto i = newValues.begin(); i != newValues.end(); i++) {
        nextFeatureValues.insert(i->find(nextFeature)->second);
    }

    Node current = Node(nextFeature, branchValue);
    string highestFrequency;
    int maximum = 0;
    for (auto a = mostFrequent.begin(); a != mostFrequent.end(); a++) {
        if (a->second > maximum) {
            maximum = a->second;
            highestFrequency = a->first;
        }
    }
    current.maxFrequencyValue = highestFrequency;
    if (parentNode.get() != nullptr) {
        current.parentNode = parentNode;
    }

    for (auto i = nextFeatureValues.begin(); i != nextFeatureValues.end(); i++) {
        vector<map<string,string>> nextStepValues;
        for (auto j = newValues.begin(); j != newValues.end(); j++) {
            map<string,string> nextFeatureNewValues;
            if ((j->find(nextFeature) == j->end()) || (j->find(nextFeature)->second != *i)) {
                continue;
            }
            for (auto k = j->begin(); k != j->end(); k++) {
                if(k->first != nextFeature) {
                    nextFeatureNewValues.insert(make_pair(k->first, k->second));
                }
            }
            nextStepValues.push_back(nextFeatureNewValues);
        }

        vector<string> nextStepFeatures;
        for (auto i : algoFeatures) {
            if (i != nextFeature) {
                nextStepFeatures.push_back(i);
            }
        }
        Node t = algorithmID3(nextStepValues, newValues, nextStepFeatures, *i, make_shared<Node>(current), depth + 1, maxDepth);
        t.parentNode = make_shared<Node>(current);
        current.childrenNodes.push_back(t);
    }
    return current;
}

void printLeafs(Node* leaf) {
    stack<Node> ancestors;
    Node* current = leaf;

    while (current != nullptr) {
        ancestors.push(*current);
        current = current->parentNode.get();
    }

    int i = 1;
    while (!ancestors.empty()) {
        if (!ancestors.top().leaf) {
            if (i == 1) {
                cout << i << ":" << ancestors.top().feature << "=";
            }
            else {
                cout << ancestors.top().branchValue << " " << i << ":"
                     << ancestors.top().feature << "=";
            }
        }
        else {
            cout << ancestors.top().branchValue << " " << ancestors.top().leafValue << endl;
        }
        i++;
        ancestors.pop();
    }
}

void printTree(Node root) {
    Node* element = &root;
    for (auto i : element->childrenNodes) {
        if (i.leaf) {
            printLeafs(&i);
        }
        else {
            printTree(i);
        }
    }
}

string prediction(Node root, map<string, string> input) {
    Node* current = &root;
    while(!current->childrenNodes.empty()) {
        bool found = false;
        for (auto i = current->childrenNodes.begin(); i != current->childrenNodes.end(); i++) {
            if (i->branchValue == input.find(current->feature)->second) {
                current = &(*i);
                found = true;
                break;
            }
        }
        if (!found) {
            return current->maxFrequencyValue;
        }
    }
    return current->leafValue;
}



int main(int argc, char* argv[]) {
    
    string pom;
    bool headerInput = false;
    ifstream file(argv[1]);

    if (file.is_open()) {
        string line;
        while(getline(file, line)) {
            pom = line;
            if (line.at(0) != '#') {
                if (!headerInput) {
                    parseInput(line, true);
                    headerInput = true;
                }
                else {
                    parseInput(line);
                }
            }
        }
    }

    ifstream file2(argv[2]);

    if (file2.is_open()) {
        string line;
        while(getline(file2, line)) {
            pom = line;
            if (line.at(0) != '#') {
                parseTrainInput(line);
            }
        }
    }
    Node rez = Node(false);
    if (argc == 3) {
        rez = algorithmID3(values, {}, header, "", nullptr);
    }
    else {
        rez = algorithmID3(values, {}, header, "", nullptr, 0, stoi(argv[3]));
    }

    set<string> freqs;
    for (auto i : values) {
        freqs.insert(i.find(label)->second);
    }

    for (auto i = freqs.begin(); i != freqs.end(); i++) {
        map<string, int> row;
        for (auto j = freqs.begin(); j != freqs.end(); j++) {
            row.insert(make_pair(*j, 0));
        }
        confusionMatrix.insert(make_pair(*i, row));
    }

    

    int same = 0;
    cout << "[PREDICTIONS]: ";
    for (auto i : trainValues) {
        map<string, int> confusionRow;
        string id3 = prediction(rez, i);
        confusionMatrix[i[label]][id3]++;
        if (id3 == i[label]) {
            same++;
        }
        cout << id3 << " ";
    }
    cout << endl;

    cout << "[BRANCHES]:" << endl;
    printTree(rez);

    double accuracy = 1.0 * same / trainValues.size();
    cout << "[ACCURACY]: ";
    cout << fixed << setprecision(5) << round(accuracy * 100000) / 100000 << endl;

    cout << "[CONFUSION_MATRIX]:" << endl;
    for (auto i = confusionMatrix.begin(); i != confusionMatrix.end(); i++) {
        bool first = true;
        for (auto j = i->second.begin(); j != i->second.end(); j++) {
            if(!first) {
                cout << " "; 
                first = false;
            }
            cout << j->second << " ";
        }
        cout << endl;
    }

}