#include "Configuration.hpp"
#include "BinaryTree.hpp"
#include "Population.hpp"

#include <iostream>
#include <string>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Invalid number of arguments" << endl;
        return 1;
    }
    Configuration config(argv[1]);
    Function function(argv[2]);

    Population population(&function, &config);
    population.generatePopulation();

    // BinaryTree tree(buildTreeMethod::GROW);

    // tree.getConfig()->printConfiguration();
    // tree.getFunction()->printFunction();

    // Node* root = new Node("1");
    // Node* left = new Node("2");
    // Node* right = new Node("3");
    // root->setLeft(left);
    // root->setRight(right);
    // Node* leftLeft = new Node("4");
    // Node* leftRight = new Node("5");
    // left->setLeft(leftLeft);
    // left->setRight(leftRight);
    // Node* rightLeft = new Node("6");
    // Node* rightRight = new Node("7");
    // right->setLeft(rightLeft);
    // right->setRight(rightRight);

    // tree.setRoot(root);

    // tree.printTree();

    population.generatePopulation();
}