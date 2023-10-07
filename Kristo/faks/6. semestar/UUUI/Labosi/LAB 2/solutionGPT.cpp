#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>

struct Literal {
    std::string name;
    bool negated;

    bool operator==(const Literal& other) const {
        return (negated == other.negated &&
                name == other.name);
    }
};

using Clause = std::vector<Literal>;
using Clauses = std::vector<Clause>;

bool parseLiteral(const std::string& token, Literal& literal) {
    if (token.empty()) return false;
    literal.negated = token[0] == '~';
    literal.name = literal.negated ? token.substr(1) : token;
    return true;
}

Clause parseClause(const std::string& line) {
    std::istringstream iss(line);
    std::vector<std::string> tokens{
        std::istream_iterator<std::string>{iss},
        std::istream_iterator<std::string>{}};

    Clause clause;
    for (const auto& token : tokens) {
        Literal literal;
        if (parseLiteral(token, literal)) {
            clause.push_back(literal);
        }
    }
    return clause;
}

Clauses parseClauses(const std::string& filename) {
    std::ifstream file(filename);
    Clauses clauses;
    std::string line;
    while (std::getline(file, line)) {
        Clause clause = parseClause(line);
        if (!clause.empty()) {
            clauses.push_back(clause);
        }
    }
    return clauses;
}

bool complementary(const Literal& a, const Literal& b) {
    return a.name == b.name && a.negated != b.negated;
}

std::pair<bool, Clause> resolve(const Clause& a, const Clause& b) {
    Clause resolvent;
    bool resolved = false;

    for (const auto& literalA : a) {
        bool found = false;
        for (const auto& literalB : b) {
            if (complementary(literalA, literalB)) {
                if (resolved) return {false, {}};
                resolved = true;
                found = true;
                break;
            }
        }
        if (!found) resolvent.push_back(literalA);
    }

    if (!resolved) return {false, {}};

    for (const auto& literalB : b) {
        bool found = false;
        for (const auto& literalA : a) {
            if (complementary(literalA, literalB)) {
                found = true;
                break;
            }
        }
        if (!found) resolvent.push_back(literalB);
    }

    return {true, resolvent};
}

bool resolution(const Clauses& clauses) {
    Clauses supportSet{clauses.back()};

    while (true) {
        Clauses newClauses;

        for (const auto& clauseA : supportSet) {
            for (const auto& clauseB : clauses) {
                auto [resolved, resolvent] = resolve(clauseA, clauseB);
                if (resolved) {
                    if (resolvent.empty()) {
                        return true;
                    }
                    newClauses.push_back(resolvent);
                }
            }
        }

        if (newClauses.empty()) {
            return false;
        }

        bool progress = false;
        for (const auto& newClause : newClauses) {
            if (std::find(supportSet.begin(), supportSet.end(), newClause) == supportSet.end()) {
                supportSet.push_back(newClause);
                progress = true;
            }
        }

        if (!progress) {
            return false;
        }
    }
}

int main(int argc, char* argv[]) {
   
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " resolution <path_to_input_file>" << std::endl;
        return 1;
    }

    std::string keyword(argv[1]);
    std::string inputFile(argv[2]);

    if (keyword != "resolution") {
        std::cerr << "Invalid keyword. Use 'resolution' to run the algorithm." << std::endl;
        return 1;
    }

    Clauses clauses = parseClauses(inputFile);

    if (clauses.empty()) {
        std::cerr << "No clauses found in the input file." << std::endl;
        return 1;
    }

    bool result = resolution(clauses);

    if (result) {
        std::cout << "Contradiction found. The goal clause can be proved." << std::endl;
    } else {
        std::cout << "No contradiction found. The goal clause cannot be proved." << std::endl;
    }

    return 0;
}
