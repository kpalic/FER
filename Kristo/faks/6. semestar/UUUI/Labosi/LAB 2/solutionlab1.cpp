#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include <map>
#include <vector>
#include <queue>
#include <stack>
#include <set>

#include <memory>
#include <algorithm>
#include <codecvt>
#include <locale> // for setting the global locale
#include <cmath>

#include <windows.h>

#pragma execution_character_set( "utf-8" )

using namespace std;

enum class searchAlgo : unsigned int {NONE, BFS, UCS, ASTAR};
enum class heuristicTask : unsigned int {NONE, CONSISTENT, OPTIMISTIC};
enum class programTask : unsigned int {ALG, H};

searchAlgo searchAlgorithm;
programTask task;
heuristicTask heuristicCheck;

string initial;
set<string> target;
map<string, vector<pair<string, double>>> succFunction;
map<string, double> heurFunction;

class Node {
    public:
    string state;
    int depth = 0;
    double priceToNode = 0;
    double estimatedPriceToGoal = 0;
    shared_ptr<Node> parentNode = nullptr;

    bool operator<(const Node& other) const {
        if (estimatedPriceToGoal != other.estimatedPriceToGoal) {
            return estimatedPriceToGoal > other.estimatedPriceToGoal;
        }
        else if (priceToNode != other.priceToNode) {
            return priceToNode > other.priceToNode;
        }
        else if (depth != other.depth) {
            return depth > other.depth;
        }
        else {
            return (state < other.state);
        }
    }

    

    bool operator==(const Node& other) const {
        return (state == other.state);
    }

    Node() {}

    Node(string state, int depth, int priceToNode, double estimatedPriceToGoal) {
        this->state = state;
        this->depth = depth;
        this->priceToNode = priceToNode;
        this->estimatedPriceToGoal = estimatedPriceToGoal;
    }

    Node(string state, int depth, int priceToNode, double estimatedPriceToGoal, shared_ptr<Node> parentNode = nullptr) {
        this->state = state;
        this->depth = depth;
        this->priceToNode = priceToNode;
        this->estimatedPriceToGoal = estimatedPriceToGoal;
        this->parentNode = parentNode;
    }
};

void printNode(Node node) {
    cout << node.state << ": \n\tdepth - " << node.depth <<
         "\n\tpriceToNode - " << node.priceToNode << 
         "\n\testimatedPrice - " << node.estimatedPriceToGoal << endl; 
}

string convertDoubleToCustomString(double input) {
    stringstream stream;
    stream << fixed << setprecision(1) << input;
    string s = stream.str();
    return s;
}

void testingApproaches() {
    Node one("A", 0, 0, 0, nullptr);
    Node two("B", 0, 0, 0, nullptr);
    Node twoB("B", 0, 1, 0, nullptr);
    Node oneDepth("A", 1, 0, 0, nullptr);
    Node onePrice("A", 0, 1, 0, nullptr);
    Node oneEstimate("A", 0, 0, 1, nullptr);
    Node oneEstimatePrice("A", 0, 1, 1, nullptr);

    // test for BFS
    // queue working as intended
    // expand vector ?
    vector<Node> expandBFS;
    expandBFS.push_back(two);
    expandBFS.push_back(one);
    cout << "Test for BFS\n";
    cout << "Expecting :\n";
    cout << "\tA(0, 0, 0)\n";
    cout << "\tB(0, 0, 0)\n";

    sort(expandBFS.begin(), expandBFS.end(),
         [](const Node& n1, const Node& n2) {
            return n1.state < n2.state;
         });

    for (auto i = expandBFS.begin(); i != expandBFS.end(); i++) {
        printNode(*i);
    }

    priority_queue<Node> openUCS;
    openUCS.push(two);
    openUCS.push(one);

    // test for UCS
    // prio queue working as intended
    cout << "\n\nTest for UCS\n";
    cout << "Expecting :\n";
    cout << "\tA(0, 0, 0)\n";
    cout << "\tA(0, 1, 0)\n";

    while (!openUCS.empty()) {
        Node pom = openUCS.top();
        printNode(pom);
        openUCS.pop();
    }

    priority_queue<Node> openASTAR;
    openASTAR.push(oneEstimate);
    openASTAR.push(one);
    openASTAR.push(oneEstimatePrice);

    // test for ASTAR
    // prio queue working as intended
    cout << "\n\nTest for ASTAR\n";
    cout << "Expecting :\n";
    cout << "\tA(0, 0, 0)\n";
    cout << "\tA(0, 0, 1)\n";
    cout << "\tA(0, 1, 1)\n";

    while (!openASTAR.empty()) {
        Node pom = openASTAR.top();
        printNode(pom);
        openASTAR.pop();
    }
}

void loadSuccesors (string path) {

    string succFile = path; 
    ifstream file(succFile);

    if (file.is_open()) {
        string line;
        string goalState;
        
        // initial
        do {
            getline(file, initial);
        } while (initial.compare("#") == 0);

        // target
        getline(file, line);
        stringstream forTarget(line);
        while(getline(forTarget, goalState, ' ')) {
            target.insert(goalState);
        }

        // successors
        // state : (nextState, cost) ...
        while (getline(file, line)) {
            string key;
            string pairs;

            // key 
            stringstream ss(line);
            getline(ss, key, ':');

            // one empty space
            getline(ss, pairs, ' ');

            // right side of successor function
            getline(ss, pairs);
            stringstream ss2(pairs);
            
            string pair;
            string stateToken;
            double costToken;
            while (getline(ss2, pair, ' ')) {
                stringstream ss3(pair);
                getline(ss3, stateToken, ',');
                ss3 >> costToken;
                succFunction[key].push_back(make_pair(stateToken, costToken));
            }
        }
    }

        // cout << "Initial : " << initial << endl;
        // cout << "Target : ";
        // for (auto i = target.begin(); i != target.end(); i++) {
        //     cout << *i << endl;
        // }

        // for (auto it = succFunction.begin(); it != succFunction.end(); it++) {
        //     cout << "\nState : " << it->first << endl;
        //     for (auto i : it->second) {
        //         cout << "=> " << i.first << ", " << i.second << endl;
        //     }
        // }
}

void loadHeuristic (string path) {
    string heurFile = path; 
    ifstream file(heurFile);

    if (file.is_open()) {
        string line;
        while(getline(file, line)) {
            stringstream ss(line);
            string key;
            getline(ss, key, ':');
            string pom;
            getline(ss, pom, ' ');
            double value;
            ss >> value;

            heurFunction.insert(make_pair(key, value));
        }
    }

    // for (auto i = heurFunction.begin(); i != heurFunction.end(); i++) {
    //     cout << "State : " << (*i).first << ", Heuristic : " << (*i).second << endl;
    // }
}

queue<Node> openBFS;
set<string> closedBFS;

vector<Node> expandBFS(Node node) {
    vector<Node> result;
    vector<pair<string, double>> successors;
    successors = succFunction[node.state];

    for (auto i : successors) {
        shared_ptr<Node> toParent = make_shared<Node>(node);
        Node ret(i.first,
                 node.depth + 1,
                 i.second + node.priceToNode,
                 0,
                 toParent);
        if (closedBFS.find(ret.state) == closedBFS.end()) {
            result.push_back(ret);
        }
    }

    sort(result.begin(), result.end(), [](const Node& a, const Node& b) {
        return a.state < b.state;
    });
    return result;
}

Node searchBFS() {
    while (!openBFS.empty()) {
        Node n = openBFS.front();
        openBFS.pop();
        closedBFS.insert(n.state);
        if (target.find(n.state) != target.end()) {
            return n;
        }
        for (auto m : expandBFS(n)) {
            if (closedBFS.find(m.state) == closedBFS.end()) {
                openBFS.push(m);
            }
        }
    }
    Node fail("FAIL", 0, 0, 0, nullptr);
    return fail;
}

priority_queue<Node> openUCS;
set<string> closedUCS;

void expandUCS(Node node) {
    vector<Node> result;
    vector<pair<string, double>> successors;
    successors = succFunction[node.state];

    for (auto i : successors) {
        shared_ptr<Node> toParent = make_shared<Node>(node);
        Node ret(i.first,
                 node.depth + 1,
                 i.second + node.priceToNode,
                 0,
                 toParent);
        if (closedBFS.find(ret.state) == closedBFS.end()) {
            openUCS.push(ret);
        }
    }
}

Node searchUCS() {
    while (!openUCS.empty()) {
        if (closedUCS.find(openUCS.top().state) != closedUCS.end()) {
            openUCS.pop();
            continue;
        }
        Node n = openUCS.top();
        openUCS.pop();
        closedUCS.insert(n.state);
        if (target.find(n.state) != target.end()) {
            return n;
        }
        expandUCS(n);
    }
    Node fail("FAIL", 0, 0, 0, nullptr);
    return fail;
}

priority_queue<Node> openASTAR;
map<string, Node> closedASTAR;

void expandASTAR(Node node) {
    vector<Node> result;
    vector<pair<string, double>> successors;
    successors = succFunction[node.state];

    // cout << "IN STATE : " << endl;
    // printNode(node);

    for (auto i : successors) {
        double heurValue = heurFunction[i.first];
        shared_ptr<Node> toParent = make_shared<Node>(node);
        Node ret(i.first,
                 node.depth + 1,
                 i.second + node.priceToNode,
                 i.second + node.priceToNode + heurValue,
                 toParent);
        // cout << "SUCC : " << endl;
        // printNode(ret);
        if (closedASTAR.find(ret.state) == closedASTAR.end()) {
            openASTAR.push(ret);
        }
        else {
            Node pom = closedASTAR.find(ret.state)->second;
            if (pom.estimatedPriceToGoal > ret.estimatedPriceToGoal) {
                closedASTAR.erase(pom.state);
                openASTAR.push(ret);
            }
        }
    }
}

Node searchASTAR() {
    while (!openASTAR.empty()) {
        if (closedASTAR.find(openASTAR.top().state) != closedASTAR.end()) {
            openASTAR.pop();
            continue;
        }
        Node n = openASTAR.top();
        openASTAR.pop();
        closedASTAR.insert(make_pair(n.state, n));
        if (target.find(n.state) != target.end()) {
            return n;
        }
        expandASTAR(n);
    }
    Node fail("FAIL", 0, 0, 0, nullptr);
    return fail;
}

string pathToNode(Node result) {
    string fullPath;
    stack<string> pathResult;
    Node* path = &result;
    pathResult.push(path->state);

    while (path != nullptr) {
        if (path->parentNode != nullptr) {
            pathResult.push(path->parentNode->state);
            path = path->parentNode.get();
        }
        else {
            path = nullptr;
        }
    }

    bool notFirst = false;
    while(!pathResult.empty()) {
        string state;
        state = pathResult.top();
        pathResult.pop();

        if (notFirst) {
            fullPath = fullPath + " => ";
        }
        
        fullPath = fullPath + state;
        notFirst = true;
    }

    return fullPath;
}

vector<pair<bool, string>> consistencyResult;

bool checkConsistency () {
    bool result = true;
    for (auto i = succFunction.begin(); i != succFunction.end(); i++) {
        string resultLineBegin = "h(" + (*i).first + ") <= h(";
        double heurOne = heurFunction[(*i).first];
        string heuristicS1 = convertDoubleToCustomString(heurOne);
        for (auto j : (*i).second) {
            bool consistent = true;
            double heurTwo = heurFunction[(j).first];
            string heuristicS2 = convertDoubleToCustomString(heurTwo);
            string resultLineEnd = resultLineBegin + j.first + ") + c: " + heuristicS1 +
                                   " <= " + heuristicS2 + " + " + convertDoubleToCustomString(j.second);
            //cout << resultLineEnd << endl;

            if (heurOne > heurTwo + j.second) {
                consistent = false;
                result = false;
            }
            consistencyResult.push_back(make_pair(consistent, resultLineEnd));
        }
    }

    sort(consistencyResult.begin(), consistencyResult.end(), 
         [](const pair<bool, string> s1, const pair<bool, string> s2) {
            return s1.second < s2.second;
        });

    // for(auto i = consistencyResult.begin(); i != consistencyResult.end(); i++) {
    //     cout << i->first << " " << i->second << endl;
    // }

    return result;
}

vector<pair<bool, string>> optimisticResult;

bool checkOptimistic() {
    bool result = true;
    for (auto i = heurFunction.begin(); i != heurFunction.end(); i++) {
        bool optimistic = true;

        Node init(i->first, 0, 0, 0, nullptr);
        openUCS.push(init);
        Node resultNode = searchUCS();
        while(!openUCS.empty()) {
            openUCS.pop();
        }
        closedUCS.clear();

        if (resultNode.priceToNode < i->second) {
            optimistic = false;
            result = false;
        }

        string lineResult = "h(" + i->first + ") <= h*: " + 
                            convertDoubleToCustomString(i->second) + " <= " +
                            convertDoubleToCustomString(resultNode.priceToNode);

        optimisticResult.push_back(make_pair(optimistic, lineResult));
    }

    sort(optimisticResult.begin(), optimisticResult.end(), 
         [](const pair<bool, string> s1, const pair<bool, string> s2) {
            return s1.second < s2.second;
        });

    return result;
}

int main (int argc, char *argv[]) {
    
    SetConsoleOutputCP( 65001 );

    if ((string)argv[3] == "--alg") {
        task = programTask::ALG;
        if ((string)argv[4] == "bfs") {
            searchAlgorithm = searchAlgo::BFS;
        }
        else if ((string)argv[4] == "ucs") {
            searchAlgorithm = searchAlgo::UCS;
        }
        else if ((string)argv[4] == "astar") {
            searchAlgorithm = searchAlgo::ASTAR;
        }
    } 
    else if ((string)argv[3] == "--h") {
        task = programTask::H;
        if ((string)argv[5] == "--check-optimistic") {
            heuristicCheck = heuristicTask::OPTIMISTIC;
        }
        else if ((string)argv[5] == "--check-consistent") {
            heuristicCheck = heuristicTask::CONSISTENT;
        }
    }

    loadSuccesors(argv[2]);

    if (task == programTask::ALG) {
        Node result;
        
        if (searchAlgorithm == searchAlgo::BFS) {
            result.state = initial;
            openBFS.push(result);
            result = searchBFS();

            bool foundSolution = false;
            if (result.state.compare("FAIL") != 0) {
                foundSolution = true;
            }
            if (foundSolution) {
            cout << "[FOUND_SOLUTION]: yes" << endl;
            cout << "[STATES VISITED]: "<< closedBFS.size() << endl;
            cout << "[PATH_LENGTH]: " << result.depth + 1 << endl;
            cout << "[TOTAL_COST]: " << result.priceToNode << ".0" << endl;
            cout << "[PATH]: " << pathToNode(result) << endl;
            }
            else {
                cout << "[FOUND_SOLUTION]: no" << endl;
            }
        }
        else if (searchAlgorithm == searchAlgo::UCS) {
            result.state = initial;
            openUCS.push(result);
            result = searchUCS();

            bool foundSolution = false;
            if (result.state.compare("FAIL") != 0) {
                foundSolution = true;
            }
            if (foundSolution) {
            cout << "[FOUND_SOLUTION]: yes" << endl;
            cout << "[STATES VISITED]: "<< closedUCS.size() << endl;
            cout << "[PATH_LENGTH]: " << result.depth + 1 << endl;
            cout << "[TOTAL_COST]: " << result.priceToNode << ".0" << endl;
            cout << "[PATH]: " << pathToNode(result) << endl;
            }
            else {
                cout << "[FOUND_SOLUTION]: no" << endl;
            }
        }
        else if (searchAlgorithm == searchAlgo::ASTAR) {
            loadHeuristic(argv[6]);
            result.state = initial;
            result.estimatedPriceToGoal = heurFunction[result.state];
            openASTAR.push(result);
            result = searchASTAR();

            bool foundSolution = false;
            if (result.state.compare("FAIL") != 0) {
                foundSolution = true;
            }
            if (foundSolution) {
            cout << "[FOUND_SOLUTION]: yes" << endl;
            cout << "[STATES VISITED]: "<< closedASTAR.size() << endl;
            cout << "[PATH_LENGTH]: " << result.depth + 1 << endl;
            cout << "[TOTAL_COST]: " << result.priceToNode << ".0" << endl;
            cout << "[PATH]: " << pathToNode(result) << endl;
            }
            else {
                cout << "[FOUND_SOLUTION]: no" << endl;
            }
        }
        
    }
    else if (task == programTask::H) {
        if (heuristicCheck == heuristicTask::CONSISTENT) {
            loadHeuristic(argv[4]);
            bool consistent = checkConsistency();
            
            for(auto i = consistencyResult.begin(); i != consistencyResult.end(); i++) {
                if (i->first) {
                    cout << "[CONDITION]: [OK] " << i->second << endl;
                }
                else {
                    cout << "[CONDITION]: [ERR] " << i->second << endl;
                }
            }

            cout << "[CONCLUSION]: Heuristic is";
            if (!consistent) {
                cout << " not";
            } 
            cout << " consistent." << endl;
        }
        else if (heuristicCheck == heuristicTask::OPTIMISTIC) {
            loadHeuristic(argv[4]);
            bool optimistic = checkOptimistic();

            for(auto i = optimisticResult.begin(); i != optimisticResult.end(); i++) {
                if (i->first) {
                    cout << "[CONDITION]: [OK] " << i->second << endl;
                }
                else {
                    cout << "[CONDITION]: [ERR] " << i->second << endl;
                }
            }

            cout << "[CONCLUSION]: Heuristic is";
            if (!optimistic) {
                cout << " not";
            } 
            cout << " optimistic." << endl;
        }
        
    }
}