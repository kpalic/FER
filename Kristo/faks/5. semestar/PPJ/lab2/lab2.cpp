#include <iostream>
#include <string>
#include <vector>
#include <stack>
#include <queue>
#include <sstream>

using namespace std;


int tablica[17][14] = {{1,0,0,0,0,0,0,0,0,1,0,0,0,1},
                       {2,3,3,3,3,3,3,3,3,2,3,3,3,3},
                       {4,0,0,0,0,0,0,0,0,5,0,0,0,0},
                       {6,0,0,0,0,0,0,0,0,0,0,0,0,0},
                       {0,0,0,0,0,0,0,0,0,7,0,0,0,0},
                       {8,8,0,8,8,0,0,8,0,0,0,0,0,0},
                       {11,10,10,9,10,10,10,10,11,11,10,11,11,11},
                       {12,12,0,12,12,0,0,12,0,0,0,0,0,0},
                       {15,14,14,15,15,13,14,14,15,15,14,15,15,15},
                       {19,20,0,16,17,0,0,18,0,0,0,0,0,0},
                       {0,0,21,0,0,0,0,0,0,0,0,0,0,0},
                       {21,0,0,0,0,0,0,0,0,0,0,0,0,0},
                       {0,0,0,0,0,0,0,0,0,0,21,0,0,0},
                       {0,0,0,0,0,0,0,0,0,0,0,21,0,0},
                       {0,0,0,0,0,0,0,0,0,0,0,0,21,0},
                       {0,0,0,0,0,0,0,0,21,0,0,0,0,0},
                       {0,0,0,0,0,0,0,0,0,0,0,0,0,22}}; 

struct node { 
   string data; 
   struct node *first; 
   struct node *second;
   struct node *third;
   struct node *fourth;
   struct node *fifth; 
   struct node *sixth;
   struct node *seventh;
   struct node *eight; 
}; 
      
   //allocates new node 
struct node* newNode(string data) { 
   // declare and allocate new node  
   struct node* node = new struct node(); 
      
   node->data = data;    // Assign data to this node
      
   // Initialize children as NULL 
   node->first = NULL; 
   node->second = NULL;
   node->third = NULL; 
   node->fourth = NULL;
   node->fifth = NULL; 
   node->sixth = NULL;
   node->seventh = NULL; 
   node->eight = NULL; 
   return(node);
}

struct node* newNode() { 
   // declare and allocate new node  
   struct node* node = new struct node(); 
      
   node->data = "";    // Assign data to this node
      
   // Initialize children as NULL 
   node->first = NULL; 
   node->second = NULL;
   node->third = NULL; 
   node->fourth = NULL;
   node->fifth = NULL; 
   node->sixth = NULL;
   node->seventh = NULL; 
   node->eight = NULL; 
   return(node);
}


stack<node*> stog;
queue<vector<string>> inputs;
queue<string> unchangedInputs;
bool accepted = false;
bool error = false;
int depth = 0;
node *program = newNode("<program>");
void preorderTree(node *head) {
   for (int i = 0; i < depth; i++) {
      cout << " ";
   }
   cout << head->data << endl;
   if (head->first != nullptr) {
      depth++;
      preorderTree(head->first);
   }
   if (head->second != nullptr) {
      depth++;
      preorderTree(head->second);
   }
   if (head->third != nullptr) {
      depth++;
      preorderTree(head->third);
   }
   if (head->fourth != nullptr) {
      depth++;
      preorderTree(head->fourth);
   }
   if (head->fifth != nullptr) {
      depth++;
      preorderTree(head->fifth);
   }
   if (head->sixth != nullptr) {
      depth++;
      preorderTree(head->sixth);
   }
   if (head->seventh != nullptr) {
      depth++;
      preorderTree(head->seventh);
   }
   if (head->eight != nullptr) {
      depth++;
      preorderTree(head->eight);
   }
   depth--;

}

int checkTable(string red, string stupac) {
   int i,j;
   if (red.compare("<program>") == 0) i = 0;
   if (red.compare("<lista_naredbi>") == 0) i = 1;
   if (red.compare("<naredba>") == 0) i = 2;
   if (red.compare("<naredba_pridruzivanja>") == 0) i = 3;
   if (red.compare("<za_petlja>") == 0) i = 4;
   if (red.compare("<E>") == 0) i = 5;
   if (red.compare("<E_lista>") == 0) i = 6;
   if (red.compare("<T>") == 0) i = 7;
   if (red.compare("<T_lista>") == 0) i = 8;
   if (red.compare("<P>") == 0) i = 9;
   if (red.compare("OP_PRIDRUZI") == 0) i = 10;
   if (red.compare("IDN") == 0) i = 11;
   if (red.compare("KR_OD") == 0) i = 12;
   if (red.compare("KR_DO") == 0) i = 13;
   if (red.compare("KR_AZ") == 0) i = 14;
   if (red.compare("D_ZAGRADA") == 0) i = 15;
   if (red.compare("PRAZAN STOG") == 0) i = 16;

   if (stupac.compare("IDN") == 0) j = 0;
   if (stupac.compare("BROJ") == 0) j = 1;
   if (stupac.compare("OP_PRIDRUZI") == 0) j = 2;
   if (stupac.compare("OP_PLUS") == 0) j = 3;
   if (stupac.compare("OP_MINUS") == 0) j = 4;
   if (stupac.compare("OP_PUTA") == 0) j = 5;
   if (stupac.compare("OP_DIJELI") == 0) j = 6;
   if (stupac.compare("L_ZAGRADA") == 0) j = 7;
   if (stupac.compare("D_ZAGRADA") == 0) j = 8;
   if (stupac.compare("KR_ZA") == 0) j = 9;
   if (stupac.compare("KR_OD") == 0) j = 10;
   if (stupac.compare("KR_DO") == 0) j = 11;
   if (stupac.compare("KR_AZ") == 0) j = 12;
   if (stupac.compare("PRAZAN NIZ") == 0) j = 13;


   //cout << "produkcija = " << tablica[i][j] << endl;
   //cout << i << " " << j << endl;
   int result = tablica[i][j];
   return result;

}

void zamjeniZadrzi(vector<string> naStog) {
   node *pom = newNode();
   pom = stog.top();
   stog.pop();
   for (int i = naStog.size(); i > 0; i--) {
      node *novo = newNode(naStog.at(i-1));
      if (i == 1) pom->first = novo;
      if (i == 2) pom->second = novo;
      if (i == 3) pom->third = novo;
      if (i == 4) pom->fourth = novo;
      if (i == 5) pom->fifth = novo;
      if (i == 6) pom->sixth = novo;
      if (i == 7) pom->seventh = novo;
      if (i == 8) pom->eight = novo;
      stog.push(novo);
   }

}

void izvuciZadrzi() {
   node *pom = newNode();
   pom = stog.top();
   pom->data = stog.top()->data;
   stog.pop();
   node *dijete = newNode("$");
   pom->first = dijete;
}

void zamjeniPomakni(vector<string> naStog) {
   node *pom = newNode();
   pom = stog.top();
   pom->data = stog.top()->data;
   stog.pop();
   for (int i = naStog.size(); i > 0; i--) {
      node *novo = newNode(naStog.at(i-1));
      if (i == 1) pom->first = novo;
      if (i == 2) pom->second = novo;
      if (i == 3) pom->third = novo;
      if (i == 4) pom->fourth = novo;
      if (i == 5) pom->fifth = novo;
      if (i == 6) pom->sixth = novo;
      if (i == 7) pom->seventh = novo;
      if (i == 8) pom->eight = novo;
      stog.push(novo);
   }
   node *pom2 = newNode();
   pom2 = stog.top();
   pom2->data = unchangedInputs.front();
   stog.pop();
   unchangedInputs.pop();
   inputs.pop();
}

void IzvuciPomakni() {
   string newData;
   inputs.pop();
   newData = unchangedInputs.front();
   unchangedInputs.pop();
   stog.top()->data = newData;
   stog.pop();
}

void primjeniProdukciju(int produkcija) {
   vector<string> pomocni;
   if (produkcija == 1) {
      pomocni.push_back("<lista_naredbi>");
      zamjeniZadrzi(pomocni);
   }

   else if (produkcija == 2) {
      pomocni.push_back("<naredba>");
      pomocni.push_back("<lista_naredbi>");
      zamjeniZadrzi(pomocni);
   }

   else if (produkcija == 3) {
      izvuciZadrzi();
   }

   else if (produkcija == 4) {
      pomocni.push_back("<naredba_pridruzivanja>");
      zamjeniZadrzi(pomocni);
   }

   else if (produkcija == 5) {
      pomocni.push_back("<za_petlja>");
      zamjeniZadrzi(pomocni);
   }

   else if (produkcija == 6) {
      pomocni.push_back("IDN");
      pomocni.push_back("OP_PRIDRUZI");
      pomocni.push_back("<E>");
      zamjeniPomakni(pomocni);
   }

   else if (produkcija == 7) {
      pomocni.push_back("KR_ZA");
      pomocni.push_back("IDN");
      pomocni.push_back("KR_OD");
      pomocni.push_back("<E>");
      pomocni.push_back("KR_DO");
      pomocni.push_back("<E>");
      pomocni.push_back("<lista_naredbi>");
      pomocni.push_back("KR_AZ");
      zamjeniPomakni(pomocni);
   }

   else if (produkcija == 8) {
      pomocni.push_back("<T>");
      pomocni.push_back("<E_lista>");
      zamjeniZadrzi(pomocni);
   }

   else if (produkcija == 9) {
      pomocni.push_back("<OP_PLUS>");
      pomocni.push_back("<E>");
      zamjeniPomakni(pomocni);
   }

   else if (produkcija == 10) {
      pomocni.push_back("<OP_MINUS>");
      pomocni.push_back("<E>");
      zamjeniPomakni(pomocni);
   }

   else if (produkcija == 11) {
      izvuciZadrzi();
   }

   else if (produkcija == 12) {
      pomocni.push_back("<P>");
      pomocni.push_back("<T_lista>");
      zamjeniZadrzi(pomocni);
   }

   else if (produkcija == 13) {
      pomocni.push_back("OP_PUTA");
      pomocni.push_back("<T>");
      zamjeniPomakni(pomocni);
   }

   else if (produkcija == 14) {
      pomocni.push_back("OP_DIJELI");
      pomocni.push_back("<T>");
      zamjeniPomakni(pomocni);
   }

   else if (produkcija == 15) {
      izvuciZadrzi();
   }

   else if (produkcija == 16) {
      pomocni.push_back("OP_PLUS");
      pomocni.push_back("<P>");
      zamjeniPomakni(pomocni);
   }

   else if (produkcija == 17) {
      pomocni.push_back("OP_MINUS");
      pomocni.push_back("<P>");
      zamjeniPomakni(pomocni);
   }

   else if (produkcija == 18) {
      pomocni.push_back("L_ZAGRADA");
      pomocni.push_back("<E>");
      pomocni.push_back("D_ZAGRADA");
      zamjeniPomakni(pomocni);
   }

   else if(produkcija == 19 || produkcija == 20) {
      node *pom = stog.top();
      node *novo = newNode(unchangedInputs.front());
      pom->first = novo;
      unchangedInputs.pop();
      stog.pop();
      inputs.pop();
   }

   else if (produkcija == 21) {
      //izvuciPomakni
      IzvuciPomakni();
   }

   else if (produkcija == 22) {
      accepted = true;
   }

   else {
      error = true;
   }
}

void sintaksniAnalizator() {
   while (!accepted && !error) {

      // preorderTree(program);
      // stack<node*> pomStog;
      // cout << "------------------" << endl;
      // while (!stog.empty()) {
      //    cout << stog.top()->data << endl;
      //    pomStog.push(stog.top());
      //    stog.pop();
      // }
      // while (!pomStog.empty()) {
      //    stog.push(pomStog.top());
      //    pomStog.pop();
      // }

      // cout << "------------------" << endl;
      // cout << inputs.front()[0] << endl;
      // cout << endl;

      string red = stog.top()->data;
      string stupac = inputs.front()[0];

      int produkcija = checkTable(red,stupac);
      primjeniProdukciju(produkcija);
      
      
      


      /* cout << red->data << endl;
      cout << stupac << endl;
      cout << "________________" << endl;
      inputs.pop();  */
      // stog.pop();
   }
}


int main() {
   
   /*
   for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 14; j++) {
         cout << tablica[i][j];
      }
      cout << endl;
   } 
   */
   
   string ulaz;
   while (getline(cin, ulaz)) {
      unchangedInputs.push(ulaz);
      stringstream X(ulaz);
      vector<string> input;
      while(getline(X, ulaz, ' ')) {
         if (!ulaz.empty()) {
            input.push_back(ulaz);
         }
      }
      inputs.push(input);
   }
      vector<string> prazan;
      prazan.push_back("PRAZAN NIZ");
      inputs.push(prazan);

      node *prazan_stog = newNode("PRAZAN STOG");
      stog.push(prazan_stog);

      stog.push(program);

      sintaksniAnalizator();

      if (error) {
         if (inputs.front()[0].compare("PRAZAN NIZ") == 0) {
            cout << "err kraj" << endl;
         }
         else {
            cout << "err " << unchangedInputs.front() << endl;
         }
      }
      else {
         preorderTree(program);
      }
}