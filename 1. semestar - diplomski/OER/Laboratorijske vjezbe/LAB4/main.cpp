#include <iostream>
#include <string>
#include <fstream>
#include <set>

#include "Node.hpp"
#include "Configuration.hpp"
#include "BinaryTree.hpp"
#include "Function.hpp"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Invalid number of arguments" << endl;
        return 1;
    }
    Configuration config(argv[1]);
    Function function(argv[2]);

    BinaryTree tree(&function, &config);


    tree.setRoot(Node::generateNode(1, config.maxTreeDepth, true, config.constantRange, nullptr, function.getVariableNames()));
    cout << "Root: " << endl;
    tree.getRoot()->printValue();
    tree.printBT("", tree.getRoot(), false);

}