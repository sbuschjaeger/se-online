#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xindex_view.hpp"

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

    xt::xarray<data_t, xt::layout_type::row_major> x_tensor = xt::xarray<data_t>::from_shape({X.size(), X[0].size()});
    xt::xarray<unsigned int> y_tensor = xt::xarray<unsigned int>::from_shape({X.size()});
    for (unsigned int i = 0; i < X.size(); ++i) {
        for (unsigned int j = 0; j < X[i].size(); ++j) {
            x_tensor(i,j) = X[i][j];
        }
        y_tensor(i) = Y[i];
    }

    return std::make_pair(x_tensor, y_tensor);
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

int main() {
    // xt::xarray<data_t, xt::layout_type::row_major> X = xt::xarray<data_t>::from_shape({1000, 10});
    // for (unsigned int i = 0; i < 1000; ++i) {
    //     for (unsigned int j = 0; j < 10; ++j) {
    //         X(i,j) = 1.0;
    //     }
    // }

    // std::cout << "X.shape() == " << xt::adapt(X.shape()) << std::endl;
    // std::vector<unsigned int> indices = {2,1}; //1190, 3593,7550

    // for (unsigned int i = 0; i < indices.size(); ++i) {
    //     std::cout << "X at INDEX " << indices[i] << ":  " << xt::row(X, indices[i]) << std::endl;
    // }
    // auto x_view = xt::view(X, xt::keep(indices), xt::all());
    // std::cout << "x_view.shape() == " << xt::adapt(x_view.shape()) << std::endl;
    // std::cout << "x_view ==  " << x_view << std::endl;

    // auto && eval_view = xt::eval(x_view);
    // std::cout << eval_view << std::endl;

    std::cout << "READING FILE " << std::endl;
    auto data = read_csv("../magic/magic04.data");
    std::cout << "DONE " << std::endl;

    auto & X = data.first;
    auto & Y = data.second;
    auto n_classes = 2;
    
    auto start = std::chrono::steady_clock::now();
    std::cout << "NORMALIZING DATA OF SHAPE " << xt::adapt(X.shape()) << std::endl;
    // Not using auto here, which is kinda important
    // std::cout << amin << std::endl;
    // std::cout << amax << std::endl;
    xt::xarray<data_t> amin = xt::amin(X, 0);
    xt::xarray<data_t> amax = xt::amax(X, 0);
    X  = (X - amin) / (amax - amin);
    // around 12 seconds
    //X  = (X - xt::amin(X, 0)) / (xt::amax(X, 0) - xt::amin(X, 0));
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> runtime_seconds = end-start;
    std::cout << "DONE TIME WAS " <<  runtime_seconds.count() << " SECONDS" << std::endl;

    std::vector<unsigned int> batch_idx(X.shape()[0]);
    std::iota(std::begin(batch_idx), std::end(batch_idx), 0); 

    unsigned int epochs = 5000;
    unsigned int batch_size = 128;

    BiasedProxEnsemble<true> est(5, n_classes, 0, 0.01, 1e-4, cross_entropy, cross_entropy_deriv);
    start = std::chrono::steady_clock::now();

    for (unsigned int i = 0; i < epochs; ++i) {
        std::random_shuffle(batch_idx.begin(), batch_idx.end());
        
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

            // Usually I would use a view here, but there seems to be a bug. See https://github.com/xtensor-stack/xtensor/issues/2240
            // auto data = xt::view(X, xt::keep(indices), xt::all());
            // auto target = xt::view(Y, xt::keep(indices), xt::all());

            xt::xarray<data_t> data = xt::xarray<data_t>::from_shape({indices.size(), (int)X.shape()[1]});
            xt::xarray<unsigned int> target = xt::xarray<data_t>::from_shape({indices.size()});
            for (unsigned int i = 0; i < indices.size(); ++i) {
                for (unsigned int j = 0; j < X.shape()[1]; ++j) {
                    data(i,j) = X(indices[i], j);
                }
                target(i) = Y(indices[i]);
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