#ifndef NODE_HPP
#define NODE_HPP

#include <iostream>
#include <variant>
#include <string>
#include <cmath>
#include <random>
#include <memory>

using namespace std;

enum class Operator {
    Plus,
    Minus,
    Multiply,
    Divide,
    Sin,
    Cos,
    Sqrt,
    Log,
    Exp
};

class Node {
    private:
        variant<double, Operator, string> value;
        Node* left;
        Node* right;
        Node* parent;

    public:
        Node() {
        }
        Node(variant<double, Operator, string> value) {
            this->value = value;
            this->left = nullptr;
            this->right = nullptr;
        }
        Node(variant<double, Operator, string> value, Node* left, Node* right, Node* parent) {
            this->value = value;
            this->left = left;
            this->right = right;
            this->parent = parent;
        }
        Node(variant<double, Operator, string> value, Node* parent) {
            this->value = value;
            this->left = NULL;
            this->right = NULL;
            this->parent = parent;
        }
        Node(Node* parent) {
            this->value = nullptr;
            this->left = nullptr;
            this->right = nullptr;
            this->parent = parent;
        }


        variant<double, Operator, string> getValue() {
            return this->value;
        }

        void setvalue(variant<double, Operator, string> value) {
            this->value = value;
        }

        void printValue() {
            if (holds_alternative<Operator>(this->value)) {
                Operator op = get<Operator>(this->value);
                switch (op) {
                    case Operator::Plus:
                        cout << "+";
                        break;
                    case Operator::Minus:
                        cout << "-";
                        break;
                    case Operator::Multiply:
                        cout << "*";
                        break;
                    case Operator::Divide:
                        cout << "/";
                        break;
                    case Operator::Sin:
                        cout << "sin";
                        break;
                    case Operator::Cos:
                        cout << "cos";
                        break;
                    case Operator::Sqrt:
                        cout << "sqrt";
                        break;
                    case Operator::Log:
                        cout << "log";
                        break;
                    case Operator::Exp:
                        cout << "exp";
                        break;
                    default:
                        cout << "unknown operator";
                        break;
                }
            }
            else if (holds_alternative<string>(this->value)) {
                cout << get<string>(this->value);
            }
            else if (holds_alternative<double>(this->value)){
                cout << get<double>(this->value);
            }
            cout << endl;
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

        double evaluate(vector<double> input) {
            double leftValue = 0;
            double rightValue = 0;
            if (this->getLeft() != nullptr) {
                leftValue = this->getLeft()->evaluate(input);
            }
            else if (this->getRight() != nullptr) {
                rightValue = this->getRight()->evaluate(input);
            }
            else {
                if (typeid(this->getValue()) == typeid(string)) {
                    int index = stoi(get<string>(this->getValue()).substr(1));
                    return input[index];
                }
                else {
                    return get<double>(this->getValue());
                }
            }

            Operator op = get<Operator>(this->getValue()); 
            double result;
            switch (op) {
                case Operator::Plus:
                    return leftValue + rightValue;
                case Operator::Minus:
                    return leftValue - rightValue;
                case Operator::Multiply:
                    return leftValue * rightValue;
                case Operator::Divide:
                    result = (rightValue == 0) ? 1 : leftValue / rightValue;
                case Operator::Sin:
                    return sin(leftValue);
                case Operator::Cos:
                    return cos(leftValue);
                case Operator::Sqrt:
                    result = (leftValue < 0) ? 1 : sqrt(leftValue);
                    return result;
                case Operator::Log:
                    result = (leftValue < 0) ? 1 : log(leftValue);
                    return result;
                case Operator::Exp:
                    return exp(leftValue);
                default:
                    return 0;
            }
        }

        static Node* generateNode(int currentDepth, int maxDepth, bool full, pair<double, double> constantRange, Node* parent, vector<string> variableNames) {
            Node* result = new Node();
            
            if (parent != nullptr) {
                result->setParent(parent);
            }

            
            random_device rd1;
            random_device rd2;
            random_device rd3;
            uniform_int_distribution<int> constantOrVariable(0, 1);
            default_random_engine constantOrVariableGenerator(rd1());
            bool variable = constantOrVariable(constantOrVariableGenerator);

            uniform_real_distribution<double> randomConstant(constantRange.first, constantRange.second);
            default_random_engine constantGenerator(rd2());
            uniform_int_distribution<int> randomOperator(0, 8);
            default_random_engine operatorGenerator(rd3());

            uniform_int_distribution<int> randomVariable(0, variableNames.size() - 1);
            default_random_engine variableGenerator;

            bool isOperator;
            bool isConstant;
            bool isVariable;

            if (currentDepth == maxDepth) {
                isOperator = false;
                isConstant = true;
                isVariable = true;
            }
            else if (full == true) {
                isOperator = true;
            }
            else {
                // it can be either operator or variable or constant
                uniform_int_distribution<int> opOrVarOrConst(0, 2);
                default_random_engine opOrVarOrConstGenerator;
                int opOrVarOrConstValue = opOrVarOrConst(opOrVarOrConstGenerator);
                if (opOrVarOrConstValue == 0) {
                    isOperator = true;
                }
                else if (opOrVarOrConstValue == 1) {
                    isVariable = true;
                }
                else {
                    isConstant = true;
                }
            }

            if (parent == nullptr) {
                //root 
                Operator op = static_cast<Operator>(randomOperator(operatorGenerator));
                result->setvalue(op);
                isOperator = true;
                isVariable = false;
                isConstant = false;
            }
            else {
                variant<double, Operator, string> value = parent->getValue();
                if (holds_alternative<Operator>(parent->getValue())) {
                    Operator op = get<Operator>(parent->getValue());
                    if (op == Operator::Cos ||
                        op == Operator::Sin ||
                        op == Operator::Sqrt ||
                        op == Operator::Log ||
                        op == Operator::Exp) {

                        //unary operator
                        if (parent->getLeft() == result) {
                            if (isOperator) {
                                Operator op = static_cast<Operator>(randomOperator(operatorGenerator));
                                result->setvalue(op);
                            }
                            else if (isVariable) {
                                double variable = randomVariable(variableGenerator);
                                result->setvalue(variableNames[variable]);
                                return result;
                            }
                            else {
                                double constant = randomConstant(constantGenerator);
                                result->setvalue(constant);
                                return result;
                            }
                        }   
                        else {
                            result = parent->getLeft();
                        }
                     }        
                }
                
            }
            if (isOperator) {
                result->setLeft(generateNode(currentDepth + 1, maxDepth, full, constantRange, result, variableNames));
                result->setRight(generateNode(currentDepth + 1, maxDepth, full, constantRange, result, variableNames));
                cout << "Tree depth: " << currentDepth << " max depth : " << maxDepth << endl;
                cout << "NOW: ";
                result->printValue();
                cout << "LEFT: ";
                result->getLeft()->printValue();
                cout << "LEFT PARENT: ";
                result->getLeft()->getParent()->printValue();
                cout << "RIGHT: ";
                result->getRight()->printValue();
                cout << "RIGHT PARENT: ";
                result->getRight()->getParent()->printValue();
            }
            return result;
        }
};


#endif