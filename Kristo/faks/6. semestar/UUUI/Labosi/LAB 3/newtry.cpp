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
#include <queue>
#include <cstring>

using namespace std;

queue<string> input;

class Node {
    public:
    string feature;
    string branchValue;
    vector<Node> childrenNodes;
    shared_ptr<Node> parentNode = nullptr;

    bool leaf;
    string leafValue;

    Node(string feature, string branchValue) : feature(feature), branchValue(branchValue) {}
    Node(bool leaf, string branchValue, string leafValue) : leaf(leaf), branchValue(branchValue), leafValue(leafValue) {}
    Node(bool leaf) : leaf(leaf) {}
};

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

int main(int argc, char* argv[]) {
    
    string pom;
    bool headerInput = false;
    ifstream file(argv[1]);

    if (file.is_open()) {
        string line;
        int rows = 0;
        while(getline(file, line)) {
            input.push(line);
            rows++;
        }
        parseInput(rows);
    }

    double testEntropy = entropy(values, header);
    cout << "ENTROPIJA\n" << testEntropy << endl << endl;

    string testString = "weather";
    double testInfGain = informationGain(values, header, testString);
    cout << "INFORMACIJSKA DOBIT\n" << testInfGain << endl << endl;

    vector<map<string, string>> testValues;
    Node rez = algorithmID3(values, {}, header, "", nullptr);
    cout << rez.feature << endl;
    printTree(rez);
    // cout << "HEADER" << endl;
    // for(auto i : header) {
    //     cout << i << endl;
    // }

    // cout << endl;
    // cout << "VALUES" << endl;
    // for(auto i : values) {
    //     for (auto j = i.begin(); j != i.end(); j++) {
    //         cout << j->first << " : " << j->second << endl;
    //     }
    //     cout << endl;
    // }
    
}