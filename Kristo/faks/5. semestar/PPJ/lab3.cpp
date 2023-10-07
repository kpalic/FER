#include <iostream>
#include <string>
#include <vector>
#include <stack>
#include <queue>
#include <sstream>
#include <map>

using namespace std;

queue<string> inputs;
bool errorQuit = false;

void leksickiAnalizator(map<string, int> declaredOutside, string last, bool inForLoop) {
    map<string, int> declaredInside;
    string current;
    string previous = last;
    string varZa;
    string linijaZa;
    while(inputs.empty() == false && errorQuit == false) {
        if(current.compare("<E>") == 0 || current.compare("<T>") == 0 ||
           current.compare("<P>") == 0 || current.compare("<E_lista>") == 0 ||
           current.compare("<T_lista>") == 0 || current.compare("$") == 0 ||
           current.compare("<lista_naredbi>") == 0) {
            
            current = inputs.front();
            inputs.pop();
            continue;
        }
        if(current.empty()) {
            current = inputs.front();
            inputs.pop();
        }
        if (current.substr(0, 5).compare("KR_ZA") == 0) {
            // KR_ZA
            map<string, int> mapToForLoop = declaredOutside;
            mapToForLoop.insert(declaredInside.begin(), declaredInside.end());
            leksickiAnalizator(mapToForLoop, current, true);

        }
        else if (current.substr(0, 3).compare("IDN") == 0) {
            stringstream ss(current);
            string lexUnit;
            string lexName;
            int lineNumber;
            string pom;

            ss >> lexUnit;
            ss >> lineNumber;
            ss >> lexName;

            //cout << current << endl;
            //cout << lineNumber << endl;

            auto cantUseThisVar = declaredInside.find(varZa);
            if(cantUseThisVar->first == lexName && cantUseThisVar->second == lineNumber) {
                errorQuit = true;
                cout << "err " << lineNumber << " " << lexName << endl;
                continue;
            }

            if (previous.substr(0, 5).compare("KR_ZA") == 0) {
                    declaredInside.insert({lexName, lineNumber});
                    varZa = lexName;
                    //cout << "upisah " << lexName << " " << lineNumber << endl;
            }
            else if(previous.substr(0, 5).compare("KR_OD") == 0 ||
                    previous.substr(0, 5).compare("KR_DO") == 0) {
                    //cout << "prethodni je KR_OD ili KR_DO" << endl;
                    if (declaredInside.find(lexName)->second == lineNumber) {
                        cout << "err " << lineNumber << " " << lexName << endl;
                        errorQuit = true;
                        continue;
                    }
                    else {
                        cout << lineNumber << " " << declaredOutside.find(lexName)->second << " " << declaredOutside.find(lexName)->first << endl;
                    }
                }
            else {
                string next;
                next = inputs.front();
                if (next.substr(0, 11).compare("OP_PRIDRUZI") == 0) {
                    // cout << "INICIJALIZACIJA UNUTAR FOR PETLJE" << endl;
                    if (inForLoop == true) {
                        if (declaredOutside.find(lexName) == declaredOutside.end() &&
                            declaredInside.find(lexName) == declaredInside.end()) {
                            declaredInside.insert({lexName, lineNumber});
                        }
                    }
                    // INICIJALIZACIJA IZVAN FOR PETLJE
                    else {
                        if (!(declaredOutside.find(lexName) != declaredOutside.end() ||
                            declaredInside.find(lexName) != declaredInside.end())) {
                            declaredOutside.insert({lexName, lineNumber});
                        }
                    }
                }
                    // KORISTENJE UNUTAR FOR PETLJE -- inicijalizirano izvan
                else if (declaredInside.find(lexName) != declaredInside.end()) {
                    cout << lineNumber << " " << declaredInside.find(lexName)->second << " " << declaredInside.find(lexName)->first << endl;
                }
                    // KORISTENJE UNUTAR FOR PETLJE -- inicijalizirano unutar
                else if (declaredOutside.find(lexName) != declaredOutside.end()){
                    cout << lineNumber << " " << declaredOutside.find(lexName)->second << " " << declaredOutside.find(lexName)->first << endl;
                }
                else {
                    cout << "err " << lineNumber << " " << lexName << endl;
                    errorQuit = true;
                }
            }
        }
        else if(current.substr(0, 5).compare("KR_AZ") == 0) {
            declaredInside.clear();
            return;
        }

        //cout << current << endl;
        previous = current;
        current = inputs.front();
        inputs.pop();
    }
}

int main() {
    string ulaz;
    while (getline(cin, ulaz)){
        inputs.push(ulaz.substr(ulaz.find_first_not_of(" "), ulaz.length()));
        //cout << ulaz.substr(ulaz.find_first_not_of(" "), ulaz.length()) << endl;
    }
    //cout <<"AAAAAAA"<< endl;
    map<string, int> emptyDeclared;
    string emptyWord;
    leksickiAnalizator(emptyDeclared, emptyWord, false);
}