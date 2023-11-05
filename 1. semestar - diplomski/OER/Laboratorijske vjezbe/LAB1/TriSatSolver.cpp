#include <iostream>
#include <random>
#include <vector>
#include <ctime>
#include <string>
#include <sstream>
#include <optional>
#include <fstream>
#include <set>



using namespace std;

class MutableBitVector;

class BitVector  {
    public:
        vector<int> bits;
        BitVector(mt19937& rand, int numberOfBits) {
            uniform_int_distribution<int> distribution(0, 1);
            for (int i = 0; i < numberOfBits; i++) {
                this->bits.push_back(distribution(rand));
            }
        }
        BitVector(vector<int> bitsList) : bits(bitsList) {}
        BitVector(int n) {
            vector<int> vector1(n, 0);
            this->bits = vector1;
        }

        int get(int index) {
            if (index >= 0 && index < bits.size()) {
                return bits.at(index);
            }
            cout << index << endl;
            throw out_of_range("Index out of range");
        }
        int getSize() {
            return bits.size();
        }
        string toString() {
            string result;
            for (int bit : bits) {
                result = result + to_string(bit);
            }
            return result;
        }

        MutableBitVector copy();
};

class MutableBitVector : public BitVector {
    public:
        MutableBitVector(vector<int> bitsList) : BitVector(bitsList) {}
        MutableBitVector(int n) : BitVector(n) {}

        void set(int index, int value) {
            if (index >= 0 && index < bits.size()) {
                bits[index] = value;
                return;
            }
            throw out_of_range("Index out of range");
        }
};

MutableBitVector BitVector::copy() {
    return MutableBitVector(bits);
}

class Clause {
    private:
        vector<int> indexes;
    
    public:
        Clause(vector<int> indexes) : indexes(indexes) {}
        int getSize() {
            return indexes.size();
        }
        int getLiteral(int index) {
            if (index >= 0 && index < indexes.size()) {
                return indexes.at(index);
            }
            throw out_of_range("Index out of range");
        }
        bool isSatisfied(BitVector assignment) {
            for (int index : indexes) {
                int inBitVector = assignment.get(abs(index) - 1);
                if (inBitVector == 1 && index > 0) {
                    return true;
                }
                else {
                    if (inBitVector == 0 && index < 0) { // index je negativan
                        return true;
                    }
                }
            }
            return false;
        }

        string toString() {
        stringstream ss;
        for (int i = 0; i < indexes.size(); ++i) {
            ss << indexes[i];
            if (i != indexes.size() - 1) {
                ss << ", ";
            }
        }
        return ss.str();
    }
};

class SATFormula {
    private:
        int numOfVariables;
        vector<Clause> clauses;

    public:
        SATFormula(int numberOfVariables, vector<Clause> clauses) 
            : numOfVariables(numberOfVariables), clauses(clauses) {} 

        int getNumberOfVariables() {
            return numOfVariables;
        }
        int getNumberOfClauses() {
            return clauses.size();
        }
        Clause getClause(int index) {
            if (index >= 0 && index < clauses.size()) {
                return clauses[index];
            }
            throw out_of_range("Index out of range");
        }
        bool isSatisfied(BitVector assignment) {
            for (Clause clause : clauses) {
                if (!clause.isSatisfied(assignment)) {
                    return false;
                }
            }
            return true;
        }

        string toString() {
            stringstream ss;
            for (Clause clause : clauses) {
                ss << clause.toString() << "\n";
            }
            return ss.str();
        }
};

class BitVectorGenerator {
    private:
        BitVector assignment;

    public:
        BitVectorGenerator(BitVector assignment) : assignment(assignment) {}

        class NeighborIterator {
            private:
                BitVector currentAssignment;
                int currentIndex;
            
            public:
                NeighborIterator(BitVector currentAssignment, int currentIndex)
                    : currentAssignment(currentAssignment), currentIndex(currentIndex) {}

                NeighborIterator& operator++() {
                    currentIndex++;
                    return *this;
                }

                bool operator!=(const NeighborIterator& other) const {
                    return currentIndex != other.currentIndex;
                }

                MutableBitVector operator*() {
                    // Logika za generiranje susjeda temeljena na currentAssignment i currentIndex
                    MutableBitVector result = currentAssignment.copy(); 
                    result.set(currentIndex, (result.get(currentIndex) == 1) ? 0 : 1);
                    return result;
                }
        };

        NeighborIterator begin() {
            return NeighborIterator(assignment, 0);
        }

        NeighborIterator end() {
            return NeighborIterator(assignment, assignment.getSize());
        }

        set<MutableBitVector> createNeighborhood() {
            set<MutableBitVector> result;
            for (int i = 0; i < assignment.getSize(); i++) {
                MutableBitVector vec = assignment.copy();
                vec.set(i, (vec.get(i) == 1) ? 0 : 1);
                result.insert(vec);
            }
            return result;
        }
};

class SATFormulaStats {
    private:
        SATFormula formula;
        optional<BitVector> currentAssignment;
        int numberOfSatisfiedClauses;
        double percentageBonus;
        vector<double> percentages;
        bool satisfied;
    public:
        SATFormulaStats(const SATFormula& formula) : 
            formula(formula), 
            numberOfSatisfiedClauses(0),
            percentageBonus(0.0), 
            satisfied(false) {}

        void setAssignment(const BitVector& assignment, bool updatePercentages) {
            currentAssignment = assignment;
            // ... Izračunajte sve potrebne pokazatelje temeljene na zadatom assignment
            // ... Ažurirajte percentages ako je updatePercentages postavljeno na true

            // Na primjer:
            // numberOfSatisfiedClauses = formula.calculateSatisfiedClauses(assignment);
            // satisfied = (numberOfSatisfiedClauses == formula.getNumberOfClauses());
            
            // Dodatna logika za izračun percentageBonus i percentages može biti dodana ovdje
        }

        int getNumberOfSatisfied() const {
            return numberOfSatisfiedClauses;
        }

        bool isSatisfied() const {
            return satisfied;
        }

        double getPercentageBonus() const {
            return percentageBonus;
        }

        double getPercentage(int index) const {
            if (index < 0 || index >= percentages.size()) {
                // Baci iznimku ili vrati neku zadani vrijednost
                return 0.0;
            }
            return percentages[index];
        }

        void reset() {
        currentAssignment = BitVector(formula.getNumberOfVariables());
        numberOfSatisfiedClauses = 0;
        percentageBonus = 0.0;
        percentages.clear();
        satisfied = false;
    }
};

class IOptAlgorithm {
public:
    virtual optional<BitVector> solve(optional<BitVector> initial) = 0;
    virtual ~IOptAlgorithm() = default;
};

class ExhaustiveSearch : public IOptAlgorithm {
    private:
    public:
        set<MutableBitVector> neighborhood;
        SATFormula formula;
        optional<BitVector> solution;

        ExhaustiveSearch(SATFormula formula, optional<BitVector> solution) : 
            formula(formula), solution(solution) {}

        
        optional<BitVector> solve(optional<BitVector> initial) {
            if (!initial.has_value()) {
                // Start with a 0-filled vector for a base case.
                BitVector baseVeca = BitVector(formula.getNumberOfVariables());
                MutableBitVector baseVec = MutableBitVector(formula.getNumberOfVariables());
                BitVectorGenerator* baseGen = new BitVectorGenerator(baseVec);
                neighborhood.insert(baseGen->createNeighborhood().begin(), baseGen->createNeighborhood().end());
                neighborhood.insert(baseVec); // Add the base vector itself as well.
            } 
            else {
                // If there's an initial solution provided, start with its neighbors.
                cout << "Initial solution provided." << endl;
                cout << "Initial solution: " << initial.value().toString() << endl;
                neighborhood.insert(initial.value().copy());  // Add the initial vector itself.
                BitVectorGenerator gen(initial.value());
                set<MutableBitVector> newNeighborhood = gen.createNeighborhood();
                neighborhood.insert(newNeighborhood.begin(), newNeighborhood.end());
                neighborhood.insert(initial.value().copy());  // Add the initial vector itself.
            }
            
            MutableBitVector current = neighborhood.begin()->bits;
            int iter = 0;
            while(!neighborhood.empty() && iter < 100000) {
                iter++;
                current = neighborhood.begin()->bits;
                // cout << "Current vector: " << current.toString() << endl;
                int last = 0;
                if (formula.isSatisfied(current)) {
                    return current;
                } 
                else {
                    BitVectorGenerator gen(current);
                    set<MutableBitVector> newNeighborhood = gen.createNeighborhood();
                    // cout << "New neighborhood: " << endl;
                    // for (MutableBitVector vec : newNeighborhood) {
                    //     cout << vec.toString() << endl;
                    // }
                    neighborhood.insert(newNeighborhood.begin(), newNeighborhood.end());
                }
                neighborhood.erase(current);
            }
            return nullopt; // No solution found.
}

};

class TriSATSolver {
public:
    static SATFormula ucitajIzDatoteke(const string& filePath) {
        ifstream inputFile(filePath);
        if (!inputFile.is_open()) {
            cerr << "Failed to open the file." << endl;
            exit(-1);
        }

        string line;
        int numVariables, numClauses;

        // Čitanje datoteke i parsiranje
        vector<Clause> clauses;
        while (getline(inputFile, line)) {
            istringstream iss(line);
            char firstChar = line[0];

            if (firstChar == 'c') continue; // zanemari komentare
            if (firstChar == 'p') {
                string temp;
                iss >> temp >> temp >> numVariables >> numClauses;
                continue;
            }

            if (firstChar == '%') break; // kraj datoteke

            vector<int> literals;
            int literal;
            while(iss >> literal) {
                if(literal == 0) break; // kraj klauzule
                literals.push_back(literal);
            }
            clauses.push_back(Clause(literals));
        }

        SATFormula formula = SATFormula(numVariables, clauses);
        cout << "Number of variables: " << formula.getNumberOfVariables() << endl;
        cout << "Number of variables: " << numVariables << endl;
        cout << "Number of clauses: " << formula.getNumberOfClauses() << endl;
        return formula;
    }

    static void main(int argc, char* argv[]) {
        if(argc != 3) {
            cerr << "Usage: TriSatSolver <algorithm_index> <file_path>" << endl;
            return;
        }

        cout << "TriSatSolver" << endl;
        cout << "-------------------" << endl;
        string base = "C:\\Users\\eaprlik\\Desktop\\FER\\1. semestar - diplomski\\OER\\Laboratorijske vjezbe\\01-3sat\\";

        cout << "Putanja do datoteke : " << base + argv[2] << endl;
        cout << "Index algoritma : " << argv[1] << endl;
        int algorithmIndex = stoi(argv[1]);
        string filePath = base + argv[2];
        SATFormula formula = TriSATSolver::ucitajIzDatoteke(filePath);
        cout << formula.toString() << endl;

        vector<int> vectorBegin(formula.getNumberOfVariables(), 0);
        cout << "vectorBegin size = " << vectorBegin.size() << endl;
        optional<BitVector> solution = BitVector(vectorBegin);

        switch(algorithmIndex) {
        case 1: {
            cout << "Initial solution provided." << endl;
            cout << "Initial solution: " << solution.value().toString() << endl;
            ExhaustiveSearch* algo = new ExhaustiveSearch(formula, solution);
            solution = algo->solve(solution);
            delete algo;
            break;
        }
        // Dodajte dodatne slučajeve za druge algoritme
        default:
            cerr << "Unknown algorithm index." << endl;
            return;
        }

        if(solution.has_value()) {
            std::cout << "Solution found: " << solution.value().toString() << std::endl;
        } 
        else {
            std::cout << "No solution found." << std::endl;
        }

        return;

        // try {
        //     SATFormula formula = ucitajIzDatoteke("neka staza");
        //     IOptAlgorithm* alg = new Algoritam1(formula /*, ... dodatni argumenti*/);
        //     optional<BitVector> solution = alg->solve({});

        //     if(solution.has_value()) {
        //         BitVector sol = solution.value();
        //         cout << "Imamo rješenje: " << sol.toString() << endl;
        //     } else {
        //         cout << "Rješenje nije pronađeno." << endl;
        //     }

        //     delete alg;
        // } catch(const exception& e) {
        //     cerr << "Dogodila se pogreška: " << e.what() << endl;
        // }
    }
};

int main(int argc, char* argv[]) {
    TriSATSolver::main(argc, argv);
    return 0;
}



// public SATFormulaStats(SATFormula formula) {...}
// analizira se predano rješenje i pamte svi relevantni pokazatelji
// primjerice, ažurira elemente polja post[...] ako drugi argument to dozvoli; računa Z; ...
// public void setAssignment(BitVector assignment, boolean updatePercentages)
// vraća temeljem onoga što je setAssignment zapamtio: broj klauzula koje su zadovoljene
// public int getNumberOfSatisfied() {...}
// // vraća temeljem onoga što je setAssignment zapamtio
// public boolean isSatisfied() {...}
// // vraća temeljem onoga što je setAssignment zapamtio: suma korekcija klauzula
// // to je korigirani Z iz algoritma 3
// public double getPercentageBonus() {...}
// // vraća temeljem onoga što je setAssignment zapamtio: procjena postotka za klauzulu
// // to su elementi polja post[...]
// public double getPercentage(int index) {...}
// // resetira sve zapamćene vrijednosti na početne (tipa: zapamćene statistike)
// public void reset() {...}
// }


// Definirajte razred BitVectorNGenerator koji predstavlja funkciju (x) i koji omogućava direktnu
// uporabu u for-petlji:
// package hr.fer.zemris.trisat;
// public class BitVectorNGenerator implements Iterable<MutableBitVector> {
// public BitVectorNGenerator(BitVector assignment) {...}
// // Vraća lijeni iterator koji na svaki next() računa sljedećeg susjeda
// @Override
// public Iterator<MutableBitVector> iterator() {...}
// // Vraća kompletno susjedstvo kao jedno polje
// public MutableBitVector[] createNeighborhood() {...}
// }
// Ovaj razred zamišljeno je da se može koristiti na sljedeći način:
// BitVectorNGenerator gen = new BitVectorNGenerator(x0);
// for(MutableBitVector n : gen) {
// radi nešto sa susjedom

// public SATFormula(int numberOfVariables, Clause[] clauses) {...}
// public int getNumberOfVariables() {...}
// public int getNumberOfClauses() {...}
// public Clause getClause(int index) {...}
// public boolean isSatisfied(BitVector assignment) {...}
// @Override
// public String toString() {...}
// }

// public Clause(int[] indexes) {...}
// vraća broj literala koji čine klauzulu
// public int getSize() {...}
// vraća indeks varijable koja je index-ti član ove klauzule
// public int getLiteral(int index) {...}
// vraća true ako predana dodjela zadovoljava ovu klauzulu
// public boolean isSatisfied(BitVector assignment) {...}
// @Override
// public String toString() {...}

// public MutableBitVector(boolean... bits) {...}
// public MutableBitVector(int n) {...}
// zapisuje predanu vrijednost u zadanu varijablu
// public void set(int index, boolean value) {...}
// public MutableBitVector copy() {...}
