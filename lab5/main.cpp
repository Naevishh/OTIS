#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <limits>
#include <fstream>
#include <string>
#include <sstream>

struct DecisionData {
    std::vector<std::vector<double>> values;
    std::vector<std::vector<int>> expertEvaluations;
};

std::istream& skipCommentsAndEmpty(std::istream& is, std::string& line) {
    while (std::getline(is, line)) {
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        if (line[start] == '#') continue;
        line = line.substr(start);
        break;
    }
    return is;
}

DecisionData readMatricesFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    DecisionData data;
    std::string line;

    if (!skipCommentsAndEmpty(file, line)) {
        throw std::runtime_error("Unexpected end of file (expected values dimensions)");
    }
    std::istringstream dimStream(line);
    size_t numAlternatives, numCriteria;
    if (!(dimStream >> numAlternatives >> numCriteria)) {
        throw std::runtime_error("Invalid values dimensions line");
    }

    data.values.resize(numAlternatives, std::vector<double>(numCriteria));
    for (size_t i = 0; i < numAlternatives; ++i) {
        if (!std::getline(file, line)) {
            throw std::runtime_error("Unexpected end of file while reading values");
        }
        std::istringstream rowStream(line);
        for (size_t j = 0; j < numCriteria; ++j) {
            if (!(rowStream >> data.values[i][j])) {
                throw std::runtime_error("Invalid value in values matrix at row " + std::to_string(i));
            }
        }
    }

    if (!skipCommentsAndEmpty(file, line)) {
        throw std::runtime_error("Unexpected end of file (expected expert dimensions)");
    }
    dimStream = std::istringstream(line);
    size_t numExperts;
    if (!(dimStream >> numExperts >> numCriteria)) {
        throw std::runtime_error("Invalid expert dimensions line");
    }

    data.expertEvaluations.resize(numExperts, std::vector<int>(numCriteria));
    for (size_t i = 0; i < numExperts; ++i) {
        if (!std::getline(file, line)) {
            throw std::runtime_error("Unexpected end of file while reading expert evaluations");
        }
        std::istringstream rowStream(line);
        for (size_t j = 0; j < numCriteria; ++j) {
            if (!(rowStream >> data.expertEvaluations[i][j])) {
                throw std::runtime_error("Invalid value in expert matrix at row " + std::to_string(i));
            }
        }
    }

    return data;
}


std::vector<double> getWeightingFactor(const std::vector<std::vector<int>>& expertEvaluations) {
    if (expertEvaluations.empty() || expertEvaluations[0].empty()) return {};

    size_t numCriteria = expertEvaluations[0].size();
    std::vector<double> sumPerCriterion(numCriteria, 0.0);

    for (const auto& expert : expertEvaluations) {
        for (size_t c = 0; c < numCriteria; ++c) {
            sumPerCriterion[c] += expert[c];
        }
    }

    double total = std::accumulate(sumPerCriterion.begin(), sumPerCriterion.end(), 0.0);
    if (total == 0.0) return std::vector<double>(numCriteria, 1.0 / numCriteria);

    std::vector<double> weights(numCriteria);
    for (size_t c = 0; c < numCriteria; ++c) {
        weights[c] = sumPerCriterion[c] / total;
    }
    return weights;
}

std::vector<double> getRelativeContribution(const std::vector<std::vector<int>>& expertEvaluations) {
    if (expertEvaluations.empty() || expertEvaluations[0].empty()) return {};

    size_t numCriteria = expertEvaluations[0].size();
    std::vector<double> sumPerCriterion(numCriteria, 0.0);

    for (const auto& expert : expertEvaluations) {
        for (size_t c = 0; c < numCriteria; ++c) {
            sumPerCriterion[c] += expert[c];
        }
    }

    std::vector<double> rc(numCriteria);
    for (size_t c = 0; c < numCriteria; ++c) {
        rc[c] = sumPerCriterion[c] / 10.0;
    }
    return rc;
}

double additiveFunction(const std::vector<double>& criterionValues, const std::vector<double>& max,
                        std::vector<double> weightingFactors){
    double result=0;
    for(size_t i=0; i<criterionValues.size(); i++){
        result+=criterionValues[i]*weightingFactors[i]/max[i];
    }
    return result;
}

double multiplicativeFunction(const std::vector<double>& criterionValues, const std::vector<double>& max,
                              std::vector<double> relativeContributions){
    double result=0;
    for(size_t i=0; i<criterionValues.size(); i++){
        result*=(1-criterionValues[i]*relativeContributions[i]/max[i]);
    }
    return 1-result;
}

std::pair<double, size_t> getPreferredSystem(const std::vector<std::vector<double>>& values,
                          const std::vector<std::vector<int>>& expertEvaluations, bool additive){
    if (values.empty() || values[0].empty()) {
        return {0,0};
    }
    size_t numAlternatives = values.size();
    size_t numCriteria = values[0].size();

    std::vector<double> max(numCriteria, std::numeric_limits<double>::lowest());
    for (size_t a = 0; a < numAlternatives; ++a) {
        for (size_t c = 0; c < numCriteria; ++c) {
            if (values[a][c] > max[c]) {
                max[c] = values[a][c];
            }
        }
    }

    std::vector<double> criteria;
    criteria.resize(numAlternatives);

    std::vector<double> weights;
    if (additive) {
        weights = getWeightingFactor(expertEvaluations);
    } else {
        weights = getRelativeContribution(expertEvaluations);
    }

    if (weights.size() != numCriteria) {
        weights.resize(numCriteria, 0.0);
    }

    for (size_t a = 0; a < numAlternatives; ++a) {
        if (additive) {
            criteria[a] = additiveFunction(values[a], max, weights);
        } else {
            criteria[a] = multiplicativeFunction(values[a], max, weights);
        }
    }
    auto best_it = std::max_element(criteria.begin(), criteria.end());
    size_t best_index = std::distance(criteria.begin(), best_it);
    double best_score = *best_it;

    return {best_score, best_index+1};
}



int main() {

    try {
        DecisionData data = readMatricesFromFile("input.txt");

        std::cout << "Values:\n";
        for (const auto& alt : data.values) {
            for (double v : alt) std::cout << v << " ";
            std::cout << "\n";
        }

        std::cout << "\nExpert evaluations:\n";
        for (const auto& exp : data.expertEvaluations) {
            for (int e : exp) std::cout << e << " ";
            std::cout << "\n";
        }

        bool additive = true;
        std::pair<double, size_t> bestScore = getPreferredSystem(data.values, data.expertEvaluations, additive);
        std::cout << "\nBest score: " << bestScore.first << "\n";
        std::cout << "\nBest alternative: " << bestScore.second << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;

}
