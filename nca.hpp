#ifndef NCA_HPP
#define NCA_HPP

#include <iostream>
#include <limits>
#include <vector>

#include <Eigen/Core>

void nearest_neighbors(const std::vector<Eigen::VectorXd>& input, const std::vector<std::string>& label) {
    unsigned int correct = 0;

    for(unsigned int i = 0; i < input.size(); ++i) {
        double min_norm = std::numeric_limits<double>::infinity();
        std::string min_norm_label;
        for(unsigned int j = 0; j < input.size(); ++j) {
            if(i == j) continue;
            double norm = (input[i] - input[j]).norm();
            if(norm < min_norm) {
                min_norm = norm;
                min_norm_label = label[j];
            }
        }

        if(label[i] == min_norm_label) {
            ++correct;
        }
    }

    std::cout << "Got " << correct << " correct out of " << input.size() << std::endl;
}

Eigen::MatrixXd scaling_matrix(const std::vector<Eigen::VectorXd>& input) {
    Eigen::MatrixXd A;
    if(input.size() == 0) return A;

    int size = input[0].size();

    std::vector< std::pair<double, double> > minmax(size, std::make_pair(std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()));
    for(std::vector<Eigen::VectorXd>::const_iterator i = input.begin(); i != input.end(); ++i) {
        for(int j = 0; j < i->size(); ++j) {
            double val = (*i)[j];
            if(val < minmax[j].first) {
                minmax[j] = std::make_pair(val, minmax[j].second);
            }
            if(val > minmax[j].second) {
                minmax[j] = std::make_pair(minmax[j].first, val);
            }
        }
    }

    A = Eigen::MatrixXd::Identity(size, size);
    for(unsigned int i = 0; i < minmax.size(); ++i) {
        A(i, i) = 1.0/(minmax[i].second - minmax[i].first);
    }

    return A;
}

std::vector<Eigen::VectorXd> scale(const Eigen::MatrixXd& ScaleA, const std::vector<Eigen::VectorXd>& input) {
    std::vector<Eigen::VectorXd> scaled_input;
    for(std::vector<Eigen::VectorXd>::const_iterator i = input.begin(); i != input.end(); ++i) {
        scaled_input.push_back(ScaleA * (*i));
    }

    return scaled_input;
}

Eigen::MatrixXd neighborhood_components_analysis(const std::vector<Eigen::VectorXd>& input, const std::vector<std::string>& label, const Eigen::MatrixXd& init, unsigned int iterations, double learning_rate) {
    Eigen::MatrixXd A = init;
    for(unsigned int it = 0; it < iterations; ++it) {
        unsigned int i = it % input.size();

        double softmax_normalization = 0.0;
        for(unsigned int k = 0; k < input.size(); ++k) {
            if(k == i) continue;
            softmax_normalization += std::exp(-(A*input[i] - A*input[k]).squaredNorm());
        }

        std::vector<double> softmax;
        for(unsigned int k = 0; k < input.size(); ++k) {
            if(k == i) softmax.push_back(0.0);
            else {
                softmax.push_back(std::exp(-(A*input[i] - A*input[k]).squaredNorm()) / softmax_normalization);
            }
        }

        double p = 0.0;
        for(unsigned int k = 0; k < softmax.size(); ++k) {
            if(label[k] == label[i]) p += softmax[k];
        }

        Eigen::MatrixXd first_term = Eigen::MatrixXd::Zero(input[0].size(), input[0].size());
        Eigen::MatrixXd second_term = Eigen::MatrixXd::Zero(input[0].size(), input[0].size());
        for(unsigned int k = 0; k < input.size(); ++k) {
            if(k == i) continue;
            Eigen::VectorXd xik = input[i] - input[k];
            Eigen::MatrixXd term = softmax[k] * (xik * xik.transpose());

            first_term += term;
            if(label[k] == label[i]) second_term += term;
        }
        first_term *= p;

        A += learning_rate*A*(first_term - second_term);
    }

    return A;
}

#endif // NCA_HPP
