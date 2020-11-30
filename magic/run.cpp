#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

#include "BiasedProxEnsemble.h"

auto read_csv(std::string const& path) {
    std::vector<std::vector<data_t>> X; 
    std::vector<unsigned int> Y; 

    std::string line;
    std::ifstream file(path);

    if (file.is_open()) {
        while (std::getline(file, line)) {
            if (line.size() > 0 && line != "\r") {
                std::vector<data_t> x;
                std::stringstream ss(line);
                std::string entry;

                // All entries are float, but the last one which is the label
                while (std::getline(ss, entry, ',')) {
                    if (entry.size() > 0) { 
                        if (entry[0] == 'g') {
                            Y.push_back(1);
                        } else if (entry[0] == 'h') {
                            Y.push_back(0);
                        } else {
                            x.push_back( static_cast<float>(atof(entry.c_str())) );
                        }
                    }
                }
                if (X.size() > 0 && x.size() != X[0].size()) {
                    std::cout << "Size mismatch detected. Ignoring line." << std::endl;
                } else {
                    X.push_back(x);
                }
            }
        }
        file.close();
    }

    return std::make_pair(X, Y);
}

std::vector<std::vector<data_t>> normalize(std::vector<std::vector<data_t>> const &data) {
    std::vector<data_t> min;
    std::vector<data_t> max;

    std::vector<std::vector<data_t>> normalized_data(data);
    for (auto const & d : data) {
        for (unsigned int i = 0; i < d.size(); ++i) {
            if (min.size() <= i) {
                min.push_back(d[i]);
            } else if (min[i] > d[i]) {
                min[i] = d[i];
            }

            if (max.size() <= i) {
                max.push_back(d[i]);
            } else if (max[i] < d[i]) {
                max[i] = d[i];
            }
        }
    }

    for (unsigned int i = 0; i < data.size(); ++i) {
        for (unsigned int j = 0; j < data[i].size(); ++j) {
            normalized_data[i][j] = (normalized_data[i][j] - min[j]) / (max[j] - min[j]);
        }
    }

    return normalized_data;
}

std::string to_string(std::vector<std::vector<data_t>> const &data) {
    std::string s;

    for (auto &x : data) {
        for (auto xi : x) {
            s += std::to_string(xi) + " ";
        }
        s += "\n";
    }

    return s;
}

void print_progress(unsigned int cur_idx, unsigned int max_idx, std::string const & pre_str, unsigned int width = 100, unsigned int precision = 4) {
    data_t progress = data_t(cur_idx) / data_t(max_idx);

    std::cout << "[" << cur_idx << "/" << max_idx << "] " << std::setprecision(precision) << pre_str <<  " " ;
    unsigned int pos = width * progress;
    for (unsigned int i = 0; i < width; ++i) {
        if (i < pos) std::cout << "█";
        //else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << " " << int(progress * 100.0) << " %\r";
    std::cout << std::flush;
}

int main() {
    xt::xarray<double> arr1{
        {1.0, 2.0, 3.0},
        {2.0, 5.0, 7.0},
        {2.0, 5.0, 7.0}
    };

    std::cout << arr1 << std::endl;

    std::cout << "READING FILE " << std::endl;
    auto data = read_csv("../magic/magic04.data");
    auto const & X = data.first;
    auto const & Y = data.second;

    auto X_normalized = normalize(X);
    auto n_classes = 2;
    // std::cout << to_string(X_normalized);
    // return 1;

    std::vector<unsigned int> batch_idx(data.first.size());
    std::iota (std::begin(batch_idx), std::end(batch_idx), 0); 

    unsigned int epochs = 5;
    unsigned int batch_size = 256;

    BiasedProxEnsemble est(15, n_classes, 0, 0.001, 1e-5);

    for (unsigned int i = 0; i < epochs; ++i) {
        std::random_shuffle(batch_idx.begin(), batch_idx.end());
        
        unsigned int cnt = 0;
        data_t loss_epoch = 0;
        data_t nonzero_epoch = 0;
        unsigned int batch_cnt = 0;
        while(cnt < batch_idx.size()) {
            std::vector<std::vector<data_t>> data;
            std::vector<unsigned int> target;

            for (unsigned int j = 0; j < batch_size; ++j) {
                if (cnt >= batch_idx.size()) {
                    break;
                } else {
                    data.push_back(X_normalized[batch_idx[cnt]]);
                    target.push_back(Y[batch_idx[cnt]]);
                    cnt += 1;
                }
            }

            auto loss = est.next(data, target);
            nonzero_epoch += est.num_trees();
            loss_epoch += loss;
            batch_cnt++;
            std::stringstream ss;
            ss << std::setprecision(4) << "loss: " << loss_epoch / (cnt * n_classes) << " nonzero: " << int(nonzero_epoch / batch_cnt);
            print_progress(cnt, batch_idx.size(), ss.str() );
        }
        std::cout << std::endl;
    }
}