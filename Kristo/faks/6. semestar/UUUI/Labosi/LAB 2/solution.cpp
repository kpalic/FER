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
    // //cout << "PFC" << endl;
    for (auto a : clauses) {
        for (auto b : a) {
            // //cout << (b.negated ? "NOT " : "") << b.name << endl;
        }
        // //cout << endl;
    }
    // //cout << "==============" << endl;
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

void parseClause2(string inputLine, bool end = false) {
    string line = inputLine;
    transform(inputLine.begin(), inputLine.end(), line.begin(), ::tolower);
    stringstream ss(line);
    set<Literal> clause;
    string atom;

    while(getline(ss, atom, ' ')) {
        if (atom != "v") {
            //cout << atom << endl;
            Literal literal;
            literal.negated = atom[0] == '~';
            literal.name = literal.negated ? atom.substr(1) : atom;
            clause.insert(literal);
        }
    }
    if (!end) {
        clauses.erase(clause);
    }
    else {
        clauses.erase(clause);
        parseFinalClause(clause);
    }
}

void parseClause(string inputLine, bool end = false) {
    string line = inputLine;
    transform(inputLine.begin(), inputLine.end(), line.begin(), ::tolower);
    stringstream ss(line);
    set<Literal> clause;
    string atom;

    while(getline(ss, atom, ' ')) {
        if (atom != "v") {
            //cout << atom << endl;
            Literal literal;
            literal.negated = atom[0] == '~';
            literal.name = literal.negated ? atom.substr(1) : atom;
            clause.insert(literal);
        }
    }
    if (!end) {
        clauses.insert(clause);
    }
    else {
        clauses.erase(clause);
        parseFinalClause(clause);
    }
}

set<set<Literal>> clausesToDelete;
void removeTautology() {
    int i = 0;
    int j = 0;
    // //cout << "TAUTOLOGIJA" << endl;

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
                        //cout << "tautologija" << endl;
                        //cout << (iOne.negated ? "~" : "") << iOne.name << endl;
                        //cout << (iTwo.negated ? "~" : "") << iTwo.name << endl;
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
    
    // //cout << "APSORPCIJA" << endl;
    for (auto a : clauses) {
        for (auto b : a) {
            // //cout << (b.negated ? "NOT " : "") << b.name << endl;
        }
        // //cout << endl;
    }
    // //cout << "==============" << endl;
    bool nilFound = false;
    int counter = 0;
    bool prosao = false;
    while (!clauses.empty() && !prosao) {
        // //cout << "KLAUZULE : "<< clauses.size() << endl;
        for (auto it : clauses) {
            if (it.size() == 0) {
                // //cout << "GEJ" << endl;
                clauses.erase(it);
                continue;
            }
            // //cout << "size it : " << it.size() << endl;
            for (auto it2 : it) {
                // //cout << (it2.negated ? "NOT " : "") << it2.name << " ";
            }
            // //cout << endl;
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
                    if (clausesToDelete.find(j) != clausesToDelete.end() ||
                        clausesToDelete.find(i) != clausesToDelete.end()) {
                            continue;
                        }
                    if (i.size() == 1) {
                        for (auto k = j.begin(); k != j.end(); k++) {
                            if (i.begin()->name.compare(k->name) == 0) {
                                //cout << "NASLI SMO JEBO GA JA " << endl;
                                if (j.size() == 1) {
                                    //cout << "EVO GA NIL MAMICU TI JEBEM" << endl;
                                    //cout << (i.begin()->negated ? "~" : "") << i.begin()->name << endl;
                                    //cout << (j.begin()->negated ? "~" : "") << j.begin()->name << endl;
                                    // //cout << "OPTION 1 :" << endl;
                                    // //cout << "left : " << (i.begin()->negated ? "~" : "") << i.begin()->name << endl;
                                    // //cout << "right : " << (k->negated ? "~" : "") << k->name << endl;
                                    // //cout << "NIL" << endl;
                                    // //cout << "============================" << endl;
                                    shared_ptr<set<Literal>> leftParent = make_shared<set<Literal>>(i);
                                    shared_ptr<set<Literal>> rightParent = make_shared<set<Literal>>(j);

                                    Literal nil{"NIL", false};
                                    nil.parentOne = leftParent;
                                    nil.parentTwo = rightParent;
                                    return nil;
                                }
                                if (i.begin()->negated == k->negated && j.size() != 1) {
                                    //cout << "BRISEMO J" << endl;
                                    //cout << (i.begin()->negated ? "~" : "") << i.begin()->name << endl;
                                    //cout << (j.begin()->negated ? "~" : "") << j.begin()->name << endl;
                                    // //cout << "OPTION 2 :" << endl;
                                    // //cout << "left : " << (i.begin()->negated ? "~" : "") << i.begin()->name << endl;
                                    // //cout << "right : ";
                                    for (auto a : j) {
                                        // //cout << (a.negated ? "~" : "") << a.name << "  ";
                                    }
                                    // //cout << endl;
                                    shared_ptr<set<Literal>> leftParent = make_shared<set<Literal>>(i);
                                    shared_ptr<set<Literal>> rightParent = make_shared<set<Literal>>(j);
                                    Literal reduction{i.begin()->name, i.begin()->negated};
                                    reduction.parentOne = leftParent;
                                    reduction.parentTwo = rightParent;
                                    newClause.insert(reduction);
                                    // //cout << (reduction.negated ? "~" : "") << reduction.name << endl;
                                    // //cout << "============================" << endl;
                                }
                                
                                if (i.begin()->negated != k->negated) {
                                    //cout << "BRISEMO K IZ J" << endl;
                                    //cout << (i.begin()->negated ? "~" : "") << i.begin()->name << endl;
                                    //cout << (j.begin()->negated ? "~" : "") << j.begin()->name << endl;
                                    eraseJot = true;
                                    // //cout << "OPTION 3 :" << endl;
                                    // //cout << "left : " << (i.begin()->negated ? "~" : "") << i.begin()->name << endl;
                                    // //cout << "right : ";
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
                                            // //cout << (a.negated ? "~" : "") << a.name << "  ";
                                        }
                                        else {
                                            // //cout << (a.negated ? "~" : "") << a.name << "  ";
                                            newPom = a;
                                        }
                                    }
                                    // //cout << endl;
                                    set<Literal> newPoml;
                                    newClauses.insert(rightParent);
                                    for (auto l : rightParent) {
                                        // //cout << (l.negated ? "~" : "") << l.name << " ";
                                    }
                                    // //cout << endl;
                                    reduction.parentTwo = make_shared<set<Literal>>(j);
                                    newClause.insert(reduction);
                                    // //cout << (reduction.negated ? "~" : "") << reduction.name << endl;
                                    // //cout << "============================" << endl;
                                }
                                //cout << (i.begin()->negated ? "~" : "") << i.begin()->name << endl;
                                // for (auto a : j) {
                                //    cout << (a.negated ? "~" : "") << a.name << endl;
                                // }
                                //cout << "==========" << endl;
                            }
                            else {
                                
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
                            // //cout << "pls kill me " << nameForCheck.name << endl;
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
    for (const auto& a : *parents) {
        cout << (a.negated ? "~" : "") << a.name << endl;
        if(a.parentOne != nullptr) {
            walkThroughElse(a.parentOne, depth + 1);
        }
        if(a.parentTwo != nullptr) {
            walkThroughElse(a.parentTwo, depth + 1);
        }
    }
}

void walkthroughResult(Literal literal, int depth = 0) {
    cout << literal.name << endl;
    walkThroughElse(literal.parentOne, depth + 1);
    walkThroughElse(literal.parentTwo, depth + 1);
}

int main (int argc, char* argv[]) {

    string pom;
    if ((string)argv[1] == "resolution") {
        ifstream file(argv[2]);
        if (file.is_open()) {
            string line;
            while(getline(file, line)) {
                pom = line;
                if (line.at(0) != '#') {
                    parseClause(line);
                }
            }
            parseClause(pom, true);
        }
        removeTautology();
        Literal rez = apsorption();
        transform(pom.begin(), pom.end(), pom.begin(), ::tolower);
        
        if (rez.name != "FAIL") {
            walkthroughResult(rez);
            cout << "[CONCLUSION]: " << pom << " is true" << endl;
        }
        else {
            cout << "[CONCLUSION]: " << pom << " is unknown" << endl;
        }
    }

    else {
        ifstream file(argv[3]);
        if (file.is_open()) {
            string line;
            while(getline(file, line)) {
                pom = line;
                if (line.at(0) != '#') {
                    if (line.at(line.length() - 1) == '+') {
                        parseClause(line.substr(0, line.length() - 2));
                    }
                    else if (line.at(line.length() - 1) == '-') {
                        parseClause2(line.substr(0, line.length() - 2));
                    }
                    else {

                        auto kopija = clauses;
                        parseClause(line.substr(0, line.length() - 2));
                        parseClause(line.substr(0, line.length() - 2), true);
                        
                        removeTautology();
                        Literal rez = apsorption();
                        transform(pom.begin(), pom.end(), pom.begin(), ::tolower);
                        
                        if (rez.name != "FAIL") {
                            walkthroughResult(rez);
                            transform(line.begin(), line.end(), line.begin(), ::tolower);
                            cout << "[CONCLUSION]: " << line.substr(0, line.length() - 2) << " is true" << endl;
                        }
                        else {
                            cout << "[CONCLUSION]: " << line.substr(0, line.length() - 2) << " is unknown" << endl;
                        }

                        clauses.clear();
                        clauses = kopija;
                    }
                    
                }
            }
            if (line.at(0) != '#') {
                    if (line.at(line.length() - 1) == '+') {
                        parseClause(line.substr(0, line.length() - 2));
                    }
                    else if (line.at(line.length() - 1) == '-') {
                        parseClause2(line.substr(0, line.length() - 2));
                    }
                    else {

                        auto kopija = clauses;
                        parseClause(line.substr(0, line.length() - 2));
                        parseClause(line.substr(0, line.length() - 2), true);
                        
                        removeTautology();
                        Literal rez = apsorption();
                        transform(pom.begin(), pom.end(), pom.begin(), ::tolower);
                        
                        if (rez.name != "FAIL") {
                            walkthroughResult(rez);
                            transform(line.begin(), line.end(), line.begin(), ::tolower);
                            cout << "[CONCLUSION]: " << line.substr(0, line.length() - 2) << " is true" << endl;
                        }
                        else {
                            cout << "[CONCLUSION]: " << line.substr(0, line.length() - 2) << " is unknown" << endl;
                        }

                        clauses.clear();
                        clauses = kopija;
                        kopija.clear();
                    }       
            }
        }
    }

    
}