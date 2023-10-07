#include <iostream>
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

#include <windows.h>

#pragma execution_character_set( "utf-8" )



using namespace std;

string initial;
set<string> target;
map<string, vector<pair<string, double>>> prijelazi;

enum class searchAlgo : unsigned int {NONE, BFS, UCS, ASTAR};
enum class heuristicTask : unsigned int {NONE, CONSISTENT, OPTIMISTIC};
enum class programTask : unsigned int {ALG, H};

searchAlgo searchAlgorithm;
programTask task;
heuristicTask heuristicCheck;


class Node {
    public:
    string state;
    int depth = 0;
    double totalPrice = 0.0;
    double heuristicValue = 0;
    double price = 0.0;
    shared_ptr<Node> parentNode;

    Node(){}


    bool operator<(const Node& other) const {

        if (totalPrice != other.totalPrice) {
            return totalPrice > other.totalPrice;
        } else {
            printf("%s------ %s \n", state, other.state );
            return (state > other.state);
        }
    }

    // bool operator>(const Node& other) const {
    //     if (totalPrice != other.totalPrice) {
    //         return totalPrice < other.totalPrice;
    //     } else {
    //         return (state.compare(other.state) > 0);
    //     }
    // }

    bool operator==(const Node& other) const {
        return (state.compare(other.state) == 0);
    }


    Node(string state, int depth, int totalPrice, shared_ptr<Node> parentNode = nullptr) {
        this->state = state;
        this->depth = depth;
        this->totalPrice = totalPrice;
        this->parentNode = parentNode;
    }

    Node(string state, int depth, int totalPrice, double heuristicValue, shared_ptr<Node> parentNode = nullptr) {
        this->state = state;
        this->depth = depth;
        this->totalPrice = totalPrice;
        this->parentNode = parentNode;
        this->heuristicValue = heuristicValue;
    }
};

queue<Node> openBFS;
set<string> closedBFS;
set<Node> closedNodesBFS;

priority_queue<Node> openUCS;
set<string> closedUCS;
set<Node> closedNodesUCS;

map<string, Node> openASTAR;
set<Node> closedASTAR;

string aStarHeuristic;
map<string, double> heuristic;



vector<Node> expandASTAR(Node& node) {
    vector<Node> result;
    vector<pair<string, double>> successors;
    successors = prijelazi[node.state];
    

    
    for (auto i : successors) {
        double heuristicForNode = heuristic[i.first];
        
        shared_ptr<Node> toParent = make_shared<Node>(node);
        Node ret(i.first, 
                node.depth + 1, 
                i.second + node.price + heuristicForNode, 
                heuristicForNode,
                toParent);
        ret.heuristicValue = heuristicForNode;
        ret.price = i.second + node.price;
        ret.totalPrice = ret.price + heuristicForNode;

        // cout << ret.state << " --> price : " << ret.price << ", total : " << 
        //     node.price + i.second + heuristicForNode << ", Heur : " << heuristicForNode << endl;

        result.push_back(ret);

    }

    return result;
}
int posjete = 0;

Node searchASTAR() {
    while(!openASTAR.empty()) {
        posjete++;
        Node min;
        min.totalPrice = 99999;
        for (auto i = openASTAR.begin(); i != openASTAR.end(); i++) {
            if (i->second.totalPrice < min.totalPrice) {
                min = i->second;
            }
        }
        Node n = min; 
        openASTAR.erase(n.state);
        if (target.find(n.state) != target.end()) {
            return n;
        }
        closedASTAR.insert(n);
        for (auto m : expandASTAR(n)) {
            bool upisi = true;
            Node oldClosed;
            bool foundOld = false;
            for (auto i = closedASTAR.begin(); i != closedASTAR.end(); i++) {
                if((*i).state.compare(m.state) == 0) {
                    oldClosed = *i;
                    foundOld = true;
                    break;
                }
            }
            if(!foundOld) {
                Node oldOpen;
                bool foundOpen = false;
                for (auto i = openASTAR.begin(); i != openASTAR.end(); i++) {
                    if((*i).first.compare(m.state) == 0) {
                    oldOpen = (*i).second;
                    foundOpen = true;
                    break;
                    }
                }
                if (!foundOpen) {
                    // prvi put igdje
                    openASTAR.insert(make_pair(m.state, m));
                }
                else {
                    // prvi put, mozda vec postoji duzi put u open
                    Node pom = oldOpen;
                    if (pom.totalPrice > m.totalPrice) {
                        openASTAR.erase(pom.state);
                        openASTAR.insert(make_pair(m.state, m));
                    }
                }
            }
            else {
                // vec je zatvoren Node ali mozda postoji kraci put
                Node pom = oldClosed;
                if (pom.totalPrice > m.totalPrice) {
                    closedASTAR.erase(pom);
                    openASTAR.erase(pom.state);
                    openASTAR.insert(make_pair(m.state, m));
                }
            } 
        }
        // cout << "\nOPEN : \n";
        // for (auto i = openASTAR.begin(); i != openASTAR.end(); i++) {
        //     cout << i->second.state << " : " << i->second.totalPrice << endl;
        // }
        // cout << "\nCLOSED : \n";
        // for (auto i = closedASTAR.begin(); i != closedASTAR.end(); i++) {
        //     cout << i->state << " : " << i->price << ", " << i->totalPrice << endl;
        // }
    }

    Node fail("FAIL", 1, 1, 0, nullptr);
    return fail;
}

vector<Node> expand(Node& node, searchAlgo algo) {
    vector<Node> result;
    vector<pair<string, double>> successors;
    successors = prijelazi[node.state];

    //bool hasOne = false;
    for (auto i : successors) {
        //hasOne = true;
        shared_ptr<Node> toParent = make_shared<Node>(node);
        Node ret(i.first, node.depth + 1, i.second + node.totalPrice, toParent);
        result.push_back(ret);
        printf("!!!!!!!!!!!!!!!\n");

        if (algo == searchAlgo::BFS) {
            sort(result.begin(), result.end(), [](const Node& a, const Node& b) {
            return a.state < b.state;
            });
        }
    }
    return result;
}

int statesVisited = 0;

Node search() {
    if (searchAlgorithm == searchAlgo::BFS) {
        while (!openBFS.empty()) {
            statesVisited++;
            Node n = openBFS.front();
            openBFS.pop();
            closedBFS.insert(n.state);
            closedNodesBFS.insert(n);
            
            if (target.find(n.state) != target.end()) {
                return n;
            }

            bool foundOldBFS = false;
            for (auto m : expand(n, searchAlgo::BFS)) {
                foundOldBFS = false;
                cout << "TU SMO \n";
                for (auto i = closedBFS.begin(); i != closedBFS.end(); i++) {
                    printf("BBBBBBBBBBBBBB\n");
                    cout << *(closedBFS.find(m.state));
                    cout << "Trazimo zatvorene --- State : " << m.state << " VS " << (*i) << endl;
                    if (m.state.compare((*i)) == 0) {
                        foundOldBFS = true;
                    }
                }
                if (!foundOldBFS) {
                    openBFS.push(m);
                    cout << "Inserted openBFS : " << m.state << " -> " << m.depth 
                         << ", " << m.totalPrice << ", " << m.parentNode->state << endl;
                }
            }
        }
    }
    else if (searchAlgorithm == searchAlgo::UCS) {
        while (!openUCS.empty()) {
            statesVisited++;
            Node n = openUCS.top();
            openUCS.pop();
            closedUCS.insert(n.state);
            closedNodesUCS.insert(n);
            
            if (target.find(n.state) != target.end()) {
                return n;
            }

            for (auto m : expand(n, searchAlgo::UCS)) {
                // cout << "Retrieved Node : " << m.state << " -> " << m.depth 
                //      << ", " << m.totalPrice << ", " << m.parentNode->state << endl;
                if (closedUCS.find(m.state) == closedUCS.end()) {
                    openUCS.push(m);
                    // cout << "Inserted openUCS : " << m.state << " -> " << m.depth 
                    //      << ", " << m.totalPrice << ", " << m.parentNode->state << endl;
                }
                       
            }
            //printf("\n\n");        
            
            // vector<Node> pom;
            // for (Node i = openUCS.top(); !openUCS.empty(); i = openUCS.top()) {
            //     openUCS.pop();
            //     pom.push_back(i);
            //     //cout << i.state << ", " << i.totalPrice << endl;
            // }
            // for (auto i = pom.begin(); i != pom.end(); i++) {
            //     if (closedUCS.find((*i).state) == closedUCS.end()) {
            //         openUCS.push(*i);
            //     }
            // }
            // printf("openUCS size : %d\n", openUCS.size());
            // for(auto i = openUCS.begin(); i != openUCS.end(); i++) {
            //     printf("state --> %s\n", i->first);
            // }
        }
    }

    // FAIL
    Node fail("FAIL", 0, 0, nullptr);
    return fail;
}

void printLine(string line) {
    cout << "Input : " << line << endl;
}

void loadInput (string path, 
                string pathHeuristic = "",
                searchAlgo sA = searchAlgo::NONE, 
                heuristicTask hT = heuristicTask::NONE, 
                programTask pT = programTask::ALG) {

    string pathToFiles = "C:\\Users\\eaprlik\\Desktop\\UUUI\\Labosi\\autograder\\data\\lab1\\files\\";
    string pathToHeuristic = pathToFiles + pathHeuristic;

    ifstream file(pathToFiles + path);

    
    if (sA == searchAlgo::BFS || sA == searchAlgo::UCS || sA == searchAlgo::ASTAR) {
        string realPathToFiles = pathToFiles + path;
        ifstream file(realPathToFiles);

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
                    prijelazi[key].push_back(make_pair(stateToken, costToken));
                }
            }
        }

        if (searchAlgorithm == searchAlgo::BFS) {
            cout << "\nBFS je \n" << endl;
        } else {
            cout << "\nUCS je \n" << endl;
        }

        // cout << "Initial : " << initial << endl;
        // cout << "Target : ";
        // for (auto i = target.begin(); i != target.end(); i++) {
        //     cout << *i << endl;
        // }

        // for (auto it = prijelazi.begin(); it != prijelazi.end(); it++) {
        //     cout << "\nState : " << it->first << endl;
        //     for (auto i : it->second) {
        //         cout << "=> " << i.first << ", " << i.second << endl;
        //     }
        // }
    }

    if (searchAlgorithm == searchAlgo::ASTAR) {
        ifstream file2(pathToHeuristic);
        string heur;

        if (file2.is_open()) {
            while(getline(file2, heur)) {
                stringstream sss(heur);
                string keyH;
                double valueH;
                string help;
                getline(sss, keyH, ':');
                getline(sss, help, ' ');
                sss >> valueH;

                heuristic.insert(make_pair(keyH, valueH));
            }
        }

        // for (auto i = heuristic.begin(); i != heuristic.end(); i++) {
        //     cout << "Heuristika -> " << i->first << ", " << i->second << endl;
        // }
    }
}

int main (int argc, char *argv[]) {

    SetConsoleOutputCP( 65001 );

    if ((string)argv[3] == "--alg") {
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
        if ((string)argv[5] == "--check-optimistic") {
            heuristicCheck = heuristicTask::OPTIMISTIC;
        }
        else if ((string)argv[5] == "--check-consistent") {
            heuristicCheck = heuristicTask::CONSISTENT;
        }
    }

    // for (int i = 0; i < argc; i++) {
    //     cout << "i: " << i << endl;
    //     cout << "arg : " << argv[i] << endl; 
    // }

    if (searchAlgorithm == searchAlgo::UCS || searchAlgorithm == searchAlgo::BFS) {
        loadInput((string)argv[2], "", searchAlgorithm, heuristicCheck, task);
    }
    else {
        loadInput((string)argv[2], (string)argv[6]);
    }

    // loadInput((string)argv[2], (string)argv[6], searchAlgorithm, heuristicCheck, task);
    printf("Input Done\n");
    Node rootNode(initial,0, 0);
    Node result;

    if (searchAlgorithm == searchAlgo::BFS) {
        openBFS.push(rootNode);
        result = search();
    } 
    else if (searchAlgorithm == searchAlgo::UCS) {
        openUCS.push(rootNode);
        result = search();
    }
    else if (searchAlgorithm == searchAlgo::ASTAR) {
        openASTAR.insert(make_pair(rootNode.state, rootNode));
        printf("# A-STAR %s\n", argv[6]);
        result = searchASTAR();
    } 
    
    // cout << "\nResult : " << result.state << " - depth : " << result.depth 
    //      << " - totalCost : " << result.totalPrice << " - parent : " << result.parentNode->state << endl;
    
    int price = 0;

    stack<string> pathResult;
    Node* path = &result;
    pathResult.push(path->state);

    vector<pair<string, double>> zaPrice = prijelazi[path->state];
    for (auto i = zaPrice.begin(); i != zaPrice.end(); i++) {
        if (i->first.compare(path->parentNode.get()->state) == 0) {
            price += i->second;
        }
    }

    while (path != nullptr) {
        if (path->parentNode != nullptr) {
            vector<pair<string, double>> zaPrice = prijelazi[path->state];
            for (auto i = zaPrice.begin(); i != zaPrice.end(); i++) {
                if (i->first.compare(path->parentNode.get()->state) == 0) {
                    price += i->second;
                }
            }
            pathResult.push(path->parentNode->state);
            path = path->parentNode.get();
        }
        else {
            path = nullptr;
        }
    }
    bool notFirst = false;
    string pathOutput;
    int pathLength = 0;
    while(!pathResult.empty()) {
        string state;
        state = pathResult.top();
        pathResult.pop();

        if (notFirst) {
            pathOutput = pathOutput + " => ";
        }
        pathLength++;
        pathOutput = pathOutput + state;
        notFirst = true;
    }

    bool foundSolution = false;
    if (result.state.compare("FAIL") != 0) {
        foundSolution = true;
    }

    if (searchAlgorithm == searchAlgo::ASTAR) {
        if (foundSolution) {
            cout << "[FOUND_SOLUTION]: yes" << endl;
        }
        else {
            cout << "[FOUND_SOLUTION]: no" << endl;
        }

        if (foundSolution) {
            cout << "[STATES VISITED]: "<< posjete << endl;
            cout << "[PATH_LENGTH]: " << pathLength << endl;
            cout << "[TOTAL_COST]: " << result.totalPrice << ".0" << endl;
            cout << "[PATH]: " << pathOutput << endl;
        }
    }

    if (searchAlgorithm == searchAlgo::BFS) {
        if (foundSolution) {
            cout << "[FOUND_SOLUTION]: yes" << endl;
        }
        else {
            cout << "[FOUND_SOLUTION]: no" << endl;
        }

        if (foundSolution) {
            cout << "[STATES VISITED]: "<< closedBFS.size() << endl;
            cout << "[PATH_LENGTH]: " << pathLength << endl;
            cout << "[TOTAL_COST]: " << result.totalPrice << ".0" << endl;
            cout << "[PATH]: " << pathOutput << endl;
        }
    }

     if (searchAlgorithm == searchAlgo::UCS) {
        if (foundSolution) {
            cout << "[FOUND_SOLUTION]: yes" << endl;
        }
        else {
            cout << "[FOUND_SOLUTION]: no" << endl;
        }

        if (foundSolution) {
            cout << "[STATES VISITED]: "<< closedUCS.size() << endl;
            cout << "[PATH_LENGTH]: " << pathLength << endl;
            cout << "[TOTAL_COST]: " << result.totalPrice << ".0" << endl;
            cout << "[PATH]: " << pathOutput << endl;
        }
    }
    
    return 0;
}