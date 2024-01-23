#ifndef BINARYTREE_HPP
#define BINARYTREE_HPP

#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <utility>
#include <fstream>
#include <sstream>
#include <chrono>
#include <random>

#include "Function.hpp"
#include "Configuration.hpp"

using namespace std;

enum buildTreeMethod {
    FULL,
    GROW
};

enum buildNodeMethod {
    TERMINAL,
    OPERATOR,
    MIXED
};

class Node {
    private:
        string value;
        Node* left;
        Node* right;
        Node* parent;
        bool isRoot;
        int depth;
    
    public:
        Node(){};

        Node(string value, Node* left, Node* right, Node* parent, int depth) {
            this->value = value;
            this->left = left;
            this->right = right;
            this->parent = parent;
            this->depth = depth;
        }

        Node(string value) {
            this->value = value;
            this->left = NULL;
            this->right = NULL;
            this->parent = NULL;
            this->depth = 0;
        }

        string getValue() {
            return this->value;
        }
        void setValue(string value) {
            this->value = value;
        }

        Node* getLeft() {
            return this->left;
        }
        void setLeft(Node* left) {
            this->left = left;
        }

        Node* getRight() {
            return this->right;
        }
        void setRight(Node* right) {
            this->right = right;
        }

        Node* getParent() {
            return this->parent;
        }
        void setParent(Node* parent) {
            this->parent = parent;
        }

        int getDepth() {
            return this->depth;
        }
        void setDepth(int depth) {
            this->depth = depth;
        }

        void printValue() {
            cout << this->value << endl;
        }

        static bool isParentUnary(string value) {
            return (value == "sin" || value == "cos" || 
                    value == "exp" || value == "log" || 
                    value == "sqrt");
        }

        static string generateNode(buildNodeMethod method, int maxDepth, pair<double, double> constantRange, set<string> operators, vector<string> variableNames) {
            
            auto now = chrono::high_resolution_clock::now();
            auto nanoseconds = chrono::duration_cast<chrono::nanoseconds>(now.time_since_epoch()).count();
            srand(static_cast<unsigned int>(nanoseconds));


            if (method == buildNodeMethod::TERMINAL) {
                // either constant or variable
                int randomIndex = rand() % 2;
                if (randomIndex == 0) {
                    //constant
                    random_device rd;
                    mt19937 gen(rd());
                    uniform_real_distribution<> dis(constantRange.first, constantRange.second);
                    double randomConstant = dis(gen);
                    return to_string(randomConstant);
                }
                else {
                    //variable
                    int randomIndex = rand() % variableNames.size();
                    return variableNames[randomIndex];
                }
            }
            else if (method == buildNodeMethod::OPERATOR) {
                //pick random operator
                int randomIndex = rand() % operators.size();
                vector<string> operatorsVector(operators.begin(), operators.end());
                return operatorsVector[randomIndex];
            }
            else {
                //generate mixed
                int randomTerminalOrOperator = rand() % 2;
                if (randomTerminalOrOperator == 0) {
                    //generate terminal
                    int randomConstOrVar = rand() % 2;
                    if (randomConstOrVar == 0) {
                        //constant
                        random_device rd;
                        mt19937 gen(rd());
                        uniform_real_distribution<> dis(constantRange.first, constantRange.second);
                        double randomConstant = dis(gen);
                        return to_string(randomConstant);
                    }
                    else {
                        //variable
                        int randomIndex = rand() % variableNames.size();
                        return variableNames[randomIndex];
                    }
                }
                else {
                    //generate operator
                    int randomOperator = rand() % operators.size();
                    vector<string> operatorsVector(operators.begin(), operators.end());
                    return operatorsVector[randomOperator];
                }
            }
        }

};

class BinaryTree {
    private:
        Node* root;
        buildTreeMethod method;

    public:
        BinaryTree(buildTreeMethod method) {
            this->method = method;
        }

        Node* getRoot() {
            return this->root;
        }
        void setRoot(Node* root) {
            this->root = root;
        }

        buildTreeMethod getMethod() {
            return this->method;
        }
        void setMethod(buildTreeMethod method) {
            this->method = method;
        }

        void printTree(const string& prefix, Node* node, bool isLeft) {
            if(node != nullptr) {
                cout << prefix;
                cout << (isLeft ? "|-- " : "'-- " );

                node->printValue();

                if(node->getLeft() != nullptr || node->getRight() != nullptr)
                {
                    printTree(prefix + (isLeft ? "|   " : "    "), node->getLeft(), true);
                }

                if(node->getRight() != nullptr)
                {
                    printTree(prefix + "    ", node->getRight(), false);
                }
            }
        }

        void printTree()
        {
            cout << "Printing tree" << endl;
            cout << "Method: " << ((this->getMethod() == buildTreeMethod::FULL) ? "FULL" : "GROW") << endl;
            printTree("", this->getRoot(), false);    
        }

        static BinaryTree* generateTree(buildTreeMethod method, int maxDepth, pair<double, double> constantRange, set<string> operators, vector<string> variableNames) {
            cout << "Generating tree" << endl;
            // generate root
            BinaryTree* tree = new BinaryTree(method);
            Node* root = new Node();
            string rootValue = Node::generateNode(buildNodeMethod::OPERATOR, maxDepth, constantRange, operators, variableNames);
            root->setValue(rootValue);
            root->setDepth(1);
            tree->setRoot(root);
            generateSubtrees(method, maxDepth, constantRange, operators, variableNames, 2, root);
            return tree;
        }

        static void generateSubtrees(buildTreeMethod method, int maxDepth, pair<double, double> constantRange, set<string> operators, vector<string> variableNames, int currentDepth = 0, Node* parent = nullptr) {
            bool isParentUnary = Node::isParentUnary(parent->getValue());
            if (method == buildTreeMethod::FULL) {
                if (currentDepth > maxDepth) {
                    string leftValue = Node::generateNode(buildNodeMethod::TERMINAL, maxDepth, constantRange, operators, variableNames);
                    Node* left = new Node(leftValue, nullptr, nullptr, parent, currentDepth);
                    parent->setLeft(left);
                    if(!isParentUnary) {
                        string rightValue = Node::generateNode(buildNodeMethod::TERMINAL, maxDepth, constantRange, operators, variableNames);
                        Node* right = new Node(rightValue, nullptr, nullptr, parent, currentDepth);
                        parent->setRight(right);
                    }
                }
                else {
                    string leftValue = Node::generateNode(buildNodeMethod::OPERATOR, maxDepth, constantRange, operators, variableNames);
                    Node* left = new Node(leftValue, nullptr, nullptr, parent, currentDepth);
                    parent->setLeft(left);
                    generateSubtrees(method, maxDepth, constantRange, operators, variableNames, currentDepth + 1, left);
                    
                    if(!isParentUnary) {
                        string rightValue = Node::generateNode(buildNodeMethod::OPERATOR, maxDepth, constantRange, operators, variableNames);
                        Node* right = new Node(rightValue, nullptr, nullptr, parent, currentDepth);
                        parent->setRight(right);
                        generateSubtrees(method, maxDepth, constantRange, operators, variableNames, currentDepth + 1, right);
                    }
                }
            }
            else {
                // if not operator, dont generate subtrees

                int randomLeft = rand() % 2;
                int randomRight = rand() % 2;
                buildNodeMethod leftMethod;
                buildNodeMethod rightMethod;

                if (randomLeft == 0) {
                    leftMethod = buildNodeMethod::OPERATOR;
                }
                else {
                    leftMethod = buildNodeMethod::TERMINAL;
                }

                if (randomRight == 0) {
                    rightMethod = buildNodeMethod::OPERATOR;
                }
                else {
                    rightMethod = buildNodeMethod::TERMINAL;
                }

                string leftValue = Node::generateNode(leftMethod, maxDepth, constantRange, operators, variableNames);
                Node* left = new Node(leftValue, nullptr, nullptr, parent, currentDepth);
                parent->setLeft(left);

                if (!isParentUnary) {
                    string rightValue = Node::generateNode(rightMethod, maxDepth, constantRange, operators, variableNames);
                    Node* right = new Node(rightValue, nullptr, nullptr, parent, currentDepth);
                    parent->setRight(right);
                }

                if (currentDepth <= maxDepth) {
                    if (leftMethod == buildNodeMethod::OPERATOR) {
                        generateSubtrees(method, maxDepth, constantRange, operators, variableNames, currentDepth + 1, left);
                    }
                    if (rightMethod == buildNodeMethod::OPERATOR && !isParentUnary) {
                        generateSubtrees(method, maxDepth, constantRange, operators, variableNames, currentDepth + 1, parent->getRight());
                    }
                }
            }
        }
};

#endif