#ifndef BINARYTREE_HPP
#define BINARYTREE_HPP

#include <iostream>
#include <string>
#include <utility>
#include <cmath>

#include "Node.hpp"
#include "Configuration.hpp"
#include "Function.hpp"

using namespace std;

class BinaryTree {
    private:
        Node* root;
        int depth = 0;
        Function* function;
        Configuration* config;
    
    public:
        BinaryTree(Function* function, Configuration* config) {
            this->function = function;
            this->config = config;
            this->root = NULL;
        }

        Node* getRoot() {
            return this->root;
        }
        void setRoot(Node* root) {
            this->root = root;
        }

        int getDepth() {
            return this->depth;
        }
        void setDepth(int depth) {
            this->depth = depth;
        }

        Function* getFunction() {
            return this->function;
        }
        void setFunction(Function* function) {
            this->function = function;
        }

        Configuration* getConfig() {
            return this->config;
        }
        void setConfig(Configuration* config) {
            this->config = config;
        }

        // void createRandomTree(int currentDepth, bool full, Node* parent) {
        //     pair<double, double> constantRange = this->config->constantRange;
        //     int maxDepth = this->config->maxTreeDepth;
        //     this->root = Node::generateNode(currentDepth, maxDepth, full, constantRange, parent, this->getFunction()->getVariableNames());
        //     if (currentDepth < maxDepth && (typeid(this->root->getValue()) == typeid(Operator))) {
        //         this->root->setLeft(createRandomTree(currentDepth + 1, full, this->root));
        //         this->root->setRight(createRandomTree(currentDepth + 1, full, this->root));
        //     }
        //     return this->root;
        // }

        void printBT(string prefix, Node* node, bool isLeft)
        {   
            if( node != nullptr )
            {
                cout << prefix;

                cout << (isLeft ? "|__" : "|--" );
                // print the value of the node
                node->getLeft()->printValue();

                // enter the next tree level - left and right branch
                if (node->getLeft() != nullptr) {
                    if (holds_alternative<Operator>(node->getLeft()->getValue())) {
                        printBT( prefix + (isLeft ? "|   " : "    "), node->getLeft(), true);
                    }
                }
                if (node->getRight() != nullptr) {
                    if (holds_alternative<Operator>(node->getRight()->getValue())) {
                        printBT( prefix + (isLeft ? "|   " : "    "), node->getRight(), false);
                    }
                }
                // printBT( prefix + (isLeft ? "| K  " : "    "), node->getLeft(), true);
                // printBT( prefix + (isLeft ? "│ R  " : "    "), node->getRight(), false);
            }
        }

        void printBT(Node* node)
        {
            printBT("", node, false);    
        }

        // pass the root node of your binary tree
        // printBT(root);
        



};

#endif