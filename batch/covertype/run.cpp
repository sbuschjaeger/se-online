#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <set>

#include "Losses.h"
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
                        if (x.size() == 54) {
                            Y.push_back( static_cast<int>(atof(entry.c_str())) - 1);
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

template <typename T>
std::string to_string(std::vector<std::vector<T>> const &data, unsigned int rows = 100) {
    std::string s;
    unsigned int cnt = 0;

    for (auto &x : data) {
        for (auto xi : x) {
            s += std::to_string(xi) + " ";
            if (++cnt >= rows) {
                break;
            }
        }
        if (cnt >= rows) {
            break;
        }
        s += "\n";
    }

    return s;
}

template <typename T>
std::string to_string(std::vector<T> const &data, unsigned int rows = 100) {
    std::string s;
    unsigned int cnt = 0;

    for (auto xi : data) {
        s += std::to_string(xi) + " ";
        if (++cnt >= rows) {
            break;
        }
    }
    s += "\n";

    return s;
}

void print_progress(unsigned int cur_idx, unsigned int max_idx, std::string const & pre_str, unsigned int width = 100, unsigned int precision = 8) {
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

auto classes(std::vector<unsigned int> const &Y) {
    std::set<unsigned int> uclasses;
    
    for (auto &yi : Y) {
        uclasses.insert(yi);
    }

    return uclasses;
}

int main() {
    std::cout << "READING FILE " << std::endl;
    auto data = read_csv("../covertype/covtype.data");
    std::cout << "DONE " << std::endl;
    auto & X = data.first;
    auto & Y = data.second;
    auto uclasses = classes(Y);
    auto n_classes = uclasses.size();
    std::cout << to_string(X,100) << std::endl;
    std::cout << to_string(Y,100) << std::endl;
    std::cout << "nclasses: " << n_classes << std::endl;
    

    auto start = std::chrono::steady_clock::now();
    std::cout << "NORMALIZING" << std::endl;
    X = normalize(X);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> runtime_seconds = end-start;
    std::cout << "DONE TIME WAS " <<  runtime_seconds.count() << " SECONDS" << std::endl;

    std::vector<unsigned int> batch_idx(X.size());
    std::iota(std::begin(batch_idx), std::end(batch_idx), 0); 

    unsigned int epochs = 2;
    unsigned int batch_size = 128;

    unsigned int max_depth = 5;
    unsigned int max_trees = 0;
    unsigned long seed = 1235;
    data_t step_size = 1e-2;
    data_t l_reg = 1e-3;
    data_t init_weight = 1.0;

    BiasedProxEnsemble<TREE_INIT::TRAIN, TREE_NEXT::GRADIENT, double> est(max_depth, max_trees, n_classes, seed, step_size, l_reg, init_weight, cross_entropy, cross_entropy_deriv);
    start = std::chrono::steady_clock::now();

    for (unsigned int i = 0; i < epochs; ++i) {
        std::shuffle(batch_idx.begin(), batch_idx.end(), std::default_random_engine(seed));
        
        unsigned int cnt = 0;
        data_t loss_epoch = 0;
        data_t nonzero_epoch = 0;
        unsigned int batch_cnt = 0;
        while(cnt < batch_idx.size()) {
            std::vector<unsigned int> indices;
            for (unsigned int j = 0; j < batch_size; ++j) {
                if (cnt >= batch_idx.size()) {
                    break;
                } else {
                    indices.push_back(batch_idx[cnt]);
                    ++cnt;
                }
            }

            std::vector<std::vector<data_t>> data(indices.size());
            std::vector<unsigned int> target(indices.size());
            for (unsigned int i = 0; i < indices.size(); ++i) {
                data[i] = X[indices[i]];
                target[i] = Y[indices[i]];
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

    end = std::chrono::steady_clock::now();   
    runtime_seconds = end-start;
    std::cout << "Runtime was " << runtime_seconds.count() << " seconds" << std::endl; 
}