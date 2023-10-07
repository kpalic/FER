#include <iostream>
#include <sstream>
#include <fstream>
#include <set>
#include <vector>
#include <algorithm>
#include <memory>
#include <cstring>

using namespace std;

class Literal {
    public:
    string name;
    bool negated;
    shared_ptr<set<Literal>> parentOne = nullptr;
    shared_ptr<set<Literal>> parentTwo = nullptr;

    bool operator==(const Literal& other) const {
        return (negated == other.negated &&
                name == other.name);
    }

    bool operator<(const Literal& other) const {
        if (negated == other.negated) {
            return name < other.name;
        }
        else {
            return negated == true;
        }
    }

    Literal() : name(""), negated(false) {};
    Literal(string name, bool negated) : name(name), negated(negated) {};
};

set<set<Literal>> clauses;

void parseFinalClause(set<Literal> final) {
    if (clauses.empty()) {
        clauses.insert(final);
    }
    for (auto i : final) {
        set<Literal> pom;
        Literal pomL = i;
        pomL.negated = pomL.negated ? false : true;
        pom.insert(pomL);
        clauses.insert(pom);
    }
}

void parseClause(string inputLine, bool end = false, bool erase = false) {
    string line = inputLine;
    transform(inputLine.begin(), inputLine.end(), line.begin(), ::tolower);
    stringstream ss(line);
    set<Literal> clause;
    string atom;

    while(getline(ss, atom, ' ')) {
        if (atom != "v") {
            // cout << atom << endl;
            Literal literal;
            literal.negated = atom[0] == '~';
            literal.name = literal.negated ? atom.substr(1) : atom;
            clause.insert(literal);
        }
    }
    if (!end && !erase) {
        clauses.insert(clause);
    }
    else if (!erase) {
        clauses.erase(clause);
        parseFinalClause(clause);
    }
    else {
        clauses.erase(clause);
    }
}

set<set<Literal>> clausesToDelete;
void removeTautology() {
    int i = 0;
    int j = 0;

    for (auto sets : clauses) {
        i = 0;
        for (auto iOne : sets) {
            j = 0;
            for (auto iTwo : sets) {
                if (i == j) {
                    continue;
                }
                else {
                    if (iOne.name == iTwo.name && iOne.negated != iTwo.negated) {
                        // tautologija
                        clausesToDelete.insert(sets);
                        // cout << "tautologija" << endl;
                        // cout << (iOne.negated ? "~" : "") << iOne.name << endl;
                        // cout << (iTwo.negated ? "~" : "") << iTwo.name << endl;
                    }
                }
                j++;
            }
            i++;
        }
    }

    for (auto a : clausesToDelete) {
        clauses.erase(a);
    }
    clausesToDelete.clear();
}

Literal apsorption() {
    
    // cout << "APSORPCIJA" << endl;
    // for (auto a : clauses) {
    //     for (auto b : a) {
    //         cout << (b.negated ? "NOT " : "") << b.name << endl;
    //     }
    //     cout << endl;
    // }
    // cout << "==============" << endl;
    bool nilFound = false;
    bool prosao = false;
    int counter = 0;
    while (!clauses.empty() && !prosao) {

        for (auto a : clauses) {
            if (a.empty()) {
                clauses.erase(a);
                continue;
            }
        }

        if (clauses.size() == 1) {
            prosao = true;
        }

        //cout << "okay!" << endl;
        set<set<Literal>> newClauses;
        for (auto i : clauses) {
            for (auto j : clauses) {
                bool eraseJot = false;
                set<Literal> newClause;
                if (i == j) {
                    continue;
                }
                else {
                    if (i.size() == 1) {
                        for (auto k = j.begin(); k != j.end(); k++) {
                            if (i.begin()->name.compare(k->name) == 0) {
                                //cout << "NASLI SMO JEBO GA JA " << endl;
                                if (j.size() == 1) {
                                    // cout << "EVO GA NIL MAMICU TI JEBEM" << endl;
                                    // cout << (i.begin()->negated ? "~" : "") << i.begin()->name << endl;
                                    // cout << (j.begin()->negated ? "~" : "") << j.begin()->name << endl;
                                    // cout << "OPTION 1 :" << endl;
                                    // cout << "left : " << (i.begin()->negated ? "~" : "") << i.begin()->name << endl;
                                    // cout << "right : " << (k->negated ? "~" : "") << k->name << endl;
                                    // cout << "NIL" << endl;
                                    // cout << "============================" << endl;
                                    shared_ptr<set<Literal>> leftParent = make_shared<set<Literal>>(i);
                                    shared_ptr<set<Literal>> rightParent = make_shared<set<Literal>>(j);

                                    Literal nil{"NIL", false};
                                    nil.parentOne = leftParent;
                                    nil.parentTwo = rightParent;
                                    nilFound = true;
                                    newClause.insert(nil);
                                    return nil;
                                }
                                if (i.begin()->negated == k->negated && j.size() != 1) {
                                    // cout << "BRISEMO J" << endl;
                                    // cout << (i.begin()->negated ? "~" : "") << i.begin()->name << endl;
                                    // cout << (j.begin()->negated ? "~" : "") << j.begin()->name << endl;
                                    // cout << "OPTION 2 :" << endl;
                                    // cout << "left : " << (i.begin()->negated ? "~" : "") << i.begin()->name << endl;
                                    // cout << "right : ";
                                    // for (auto a : j) {
                                    //     cout << (a.negated ? "~" : "") << a.name << "  ";
                                    // }
                                    // cout << endl;
                                    shared_ptr<set<Literal>> leftParent = make_shared<set<Literal>>(i);
                                    shared_ptr<set<Literal>> rightParent = make_shared<set<Literal>>(j);
                                    Literal reduction{i.begin()->name, i.begin()->negated};
                                    reduction.parentOne = leftParent;
                                    reduction.parentTwo = rightParent;
                                    newClause.insert(reduction);
                                    // cout << (reduction.negated ? "~" : "") << reduction.name << endl;
                                    // cout << "============================" << endl;
                                }
                                
                                if (i.begin()->negated != k->negated) {
                                    // cout << "BRISEMO K IZ J" << endl;
                                    // cout << (i.begin()->negated ? "~" : "") << i.begin()->name << endl;
                                    // cout << (j.begin()->negated ? "~" : "") << j.begin()->name << endl;
                                    // cout << "OPTION 3 :" << endl;
                                    // cout << "left : " << (i.begin()->negated ? "~" : "") << i.begin()->name << endl;
                                    // cout << "right : ";
                                    eraseJot = true;
                                    shared_ptr<set<Literal>> leftParent = make_shared<set<Literal>>(i);
                                    // vise literala se vraća

                                    Literal reduction{i.begin()->name, i.begin()->negated};
                                    reduction.parentOne = leftParent;

                                    set<Literal> rightParent;
                                    Literal newPom;
                                    for (auto a : j) {
                                        if (a.name != k->name) {
                                            Literal inPom;
                                            inPom.name = a.name;
                                            inPom.negated = a.negated;
                                            inPom.parentOne = leftParent;
                                            inPom.parentTwo = make_shared<set<Literal>>(j);
                                            rightParent.insert(inPom);
                                            // cout << (a.negated ? "~" : "") << a.name << "  ";
                                        }
                                        else {
                                            // cout << (a.negated ? "~" : "") << a.name << "  ";
                                            newPom = a;
                                        }
                                    }
                                    // cout << endl;
                                    set<Literal> newPoml;
                                    newClauses.insert(rightParent);
                                    reduction.parentTwo = make_shared<set<Literal>>(j);
                                    newClause.insert(reduction);
                                    // cout << (reduction.negated ? "~" : "") << reduction.name << endl;
                                    // cout << "============================" << endl;
                                }
                                // cout << (i.begin()->negated ? "~" : "") << i.begin()->name << endl;
                                // for (auto a : j) {
                                //     cout << (a.negated ? "~" : "") << a.name << endl;
                                // }
                                // cout << "==========" << endl;
                            }
                        newClauses.insert(newClause);
                        }
                    }
                }
                if (eraseJot) {
                    clausesToDelete.insert(j);
                }
            }
        }
        for (const auto& a : newClauses) {
            clauses.insert(a);
            //cout << "wtf" << endl;
        }
        for (auto b : clausesToDelete) {
            clauses.erase(b);
        }
        counter++;
        clausesToDelete.clear();
        if (counter % 10 == 0) {
            int innerCount = 0;
            for (auto m : clauses) {
                    for (auto n : m) {
                        Literal nameForCheck = n;
                        for (auto o : clauses) {
                            for (auto p : o) {
                                if (p.name == nameForCheck.name) {
                                    innerCount++;
                                }
                                if (innerCount > 1 ) {
                                    continue;
                                }
                            }
                            if (innerCount == 1) {
                                clausesToDelete.insert(m);
                            }
                        }
                        innerCount = 0;
                    }
            }
            for (auto killMe : clausesToDelete) {
                clauses.erase(killMe);
                counter = 0;
            }
        }
    }
    Literal fail{"FAIL", false};
    return fail;
}

void walkThroughElse(shared_ptr<set<Literal>> parents, int depth) {
    bool first = true;
    cout << endl;
    for (int i = 0; i != depth; i++) {
        cout << "-------------";
    }
    for (const auto& a : *parents) {
            cout << (a.negated ? "~" : "") << a.name << " ";
    }
    if(parents.get()->begin()->parentOne != nullptr) {
        cout << "\nLeft parent : ";
        walkThroughElse(parents->begin()->parentOne, depth + 1);
    }
    if(parents.get()->begin()->parentTwo != nullptr) {
        cout << "\nRight parent : ";
        walkThroughElse(parents->begin()->parentOne, depth + 1);
    }
}

void walkthroughResult(shared_ptr<set<Literal>> clause, int depth = 0) {
    for (int i = 0; i < depth; i++) {
        cout << "---";
    }
    
    for (auto j = clause.get()->begin(); j != clause.get()->end(); j++) {
        cout << (j->negated ? "~" : "") << j->name << " ";
    }
    cout << endl;
    cout << "\nLeft parent : ";
    walkthroughResult(clause->begin()->parentOne, depth + 1);
    cout << "\nRight parent : ";
    walkthroughResult(clause->begin()->parentOne, depth + 1);
}

void parseInput(string one) {
    string pom;
    ifstream file(one);
    if (file.is_open()) {
        string line;
        while(getline(file, line)) {
            pom = line;
            //cout << line << endl;
            if (line.at(0) != '#') {
                parseClause(line);
            }
        }
        parseClause(pom, true);
    }
}

void upitnik(string line, string file) {
    string pom = line;
    line = line.substr(0, line.length() - 2);
    parseClause(line);
    parseClause(line, true);
    removeTautology();
    Literal rez = apsorption();
    transform(pom.begin(), pom.end(), pom.begin(), ::tolower);

    if (rez.name != "FAIL") {
        set<Literal> pom2;
        pom2.insert(rez);
        shared_ptr<set<Literal>> walk = make_shared<set<Literal>>(pom2);
        //walkthroughResult(walk);
        cout << "[CONCLUSION]: " << pom.substr(0, pom.length() - 2) << " is true" << endl;
    }
    else {
        cout << "[CONCLUSION]: " << pom.substr(0, pom.length() - 2) << " is unknown" << endl;
    }
    clauses.clear();
    parseInput(file);
}

int main (int argc, char* argv[]) {

    string pom;
    if ((string)argv[1] == "resolution") {
        ifstream file(argv[2]);
        if (file.is_open()) {
            string line;
            while(getline(file, line)) {
                pom = line;
                //cout << line << endl;
                if (line.at(0) != '#') {
                    parseClause(line);
                }
            }
            parseClause(pom, true);
        }
        if ((string)argv[1] == "resolution") {
        // for (auto a : clauses) {
            // for (auto b : a) {
            //     cout << (b.negated ? "NOT " : "") << b.name << endl;
            // }
            // cout << endl;
            // }   
        }
        //cout << endl;
        removeTautology();
        Literal rez = apsorption();
        transform(pom.begin(), pom.end(), pom.begin(), ::tolower);
    
        if (rez.name != "FAIL") {
            set<Literal> pom2;
            pom2.insert(rez);
            shared_ptr<set<Literal>> walk = make_shared<set<Literal>>(pom2);
            //walkthroughResult(walk);
            cout << "[CONCLUSION]: " << pom << " is true" << endl;
        }
        else {
            cout << "[CONCLUSION]: " << pom << " is unknown" << endl;
        }
    }
    else {
        // kuharica 
        string kuharica;
        string pom;
        ifstream file(argv[3]);
        if (file.is_open()) {
            string line;
            while(getline(file, line)) {
                pom = line;
                //cout << line << endl;
                if (line.at(0) != '#') {
                    if(line.at(line.length() - 1) == '?') {
                        upitnik(line, (string)argv[2]);
                        cout << "==============" << endl;
                        cout << line << endl;
                        for (auto a : clauses) {
                            for (auto b : a) {
                                cout << (b.negated ? "NOT " : "") << b.name << endl;
                            }
                            cout << endl;
                        }   
                    }
                    else if (line.at(line.length() - 1) == '+') {
                        line = line.substr(0, line.length() - 2);
                        parseClause(line);
                        cout << "==============" << endl;
                        cout << line << endl;
                        for (auto a : clauses) {
                            for (auto b : a) {
                                cout << (b.negated ? "NOT " : "") << b.name << endl;
                            }
                            cout << endl;
                        }   
                    }
                    else if (line.at(line.length() - 1) == '-') {
                        line = line.substr(0, line.length() - 2);
                        parseClause(line, false, true);
                        cout << "==============" << endl;
                        cout << line << endl;
                        for (auto a : clauses) {
                            for (auto b : a) {
                                cout << (b.negated ? "NOT " : "") << b.name << endl;
                            }
                            cout << endl;
                        }   
                    }
                }
            }
            if(pom.at(pom.length() - 1) == '?') {
                upitnik(pom, (string)argv[2]);
            }
            else if (pom.at(pom.length() - 1) == '+') {
                pom = pom.substr(0, pom.length() - 2);
                parseClause(pom);
                cout << "==============" << endl;
                cout << pom << endl;
                for (auto a : clauses) {
                    for (auto b : a) {
                        cout << (b.negated ? "NOT " : "") << b.name << endl;
                    }
                    cout << endl;
                }   
            }
            else if (pom.at(pom.length() - 1) == '-') {
                pom = pom.substr(0, pom.length() - 2);
                parseClause(pom, false, true);
                cout << "==============" << endl;
                cout << pom << endl;
                for (auto a : clauses) {
                    for (auto b : a) {
                        cout << (b.negated ? "NOT " : "") << b.name << endl;
                    }
                    cout << endl;
                }   
            }
        }
    }

    

    // if ((string)argv[1] == "resolution") {
    //     for (auto a : clauses) {
    //     for (auto b : a) {
    //         cout << (b.negated ? "NOT " : "") << b.name << endl;
    //     }
    //     cout << endl;
    //     }   
    // }
}