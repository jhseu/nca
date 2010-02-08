#include "nca.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include <boost/tokenizer.hpp>

#include <Eigen/Core>

typedef boost::tokenizer< boost::escaped_list_separator<char> > CsvSplitter;

// Counts how many columns there are in the string
unsigned int csv_size(const std::string& s) {
    CsvSplitter tok(s);
    int size = -1;
    for(CsvSplitter::const_iterator t = tok.begin(); t != tok.end(); ++t) ++size;
    if(size <= 0) {
        std::cerr << "Error: No columns available" << std::endl;
        exit(1);
    }
    return size;
}

int main(int argc, char *argv[]) {
    if(argc < 5) {
        std::cerr << "usage: " << argv[0] << " csv_file label_index iterations learning_rate" << std::endl << std::endl;
        std::cerr << "\tcsv_file: the input file. It must be in csv format." << std::endl;
        std::cerr << "\tlabel_index: the index (starting from 0) of the class label on each line of the input file." << std::endl;
        std::cerr << "\titerations: how many iterations to run sgd. Make this at least 100000 for good results." << std::endl;
        std::cerr << "\tlearning_rate: how fast it learns. 0.01 or 0.001 is usually a good value." << std::endl;
        exit(1);
    }

    std::ifstream csv_file(argv[1]);
    int label_index = std::atoi(argv[2]);
    int iterations = std::atoi(argv[3]);
    double learning_rate = std::atof(argv[4]);

    // Input parsing
    std::vector<Eigen::VectorXd> input;
    std::vector<std::string> label;
    std::string s;

    int size = -1;
    while(std::getline(csv_file, s)) {
        if(size < 0) size = csv_size(s);

        CsvSplitter tok(s);
        Eigen::VectorXd x(size);
        int i = 0, pos = 0;
        for(CsvSplitter::const_iterator t = tok.begin(); t != tok.end(); ++t) {
            if(pos == label_index) {
                label.push_back(*t);
            } else {
                x[i] = std::atof(t->c_str());
                ++i;
            }
            ++pos;
        }

        input.push_back(x);
    }

    // Test
    Eigen::MatrixXd A = neighborhood_components_analysis(input, label, scaling_matrix(input), iterations, learning_rate);

    std::vector<Eigen::VectorXd> nca_input(scale(A, input));

    std::cout << A << std::endl << std::endl;

    std::cout << "Nearest neighbors on raw data:" << std::endl;
    nearest_neighbors(input, label);

    std::cout << std::endl << "Nearest neighbors on scaled data:" << std::endl;
    nearest_neighbors(scale(scaling_matrix(input), input), label);

    std::cout << std::endl << "Nearest neighbors on nca data:" << std::endl;
    nearest_neighbors(nca_input, label);

    return 0;
}
