#ifndef TREE_H
#define TREE_H

#include <vector>
#include <math.h>
#include <random>
#include <algorithm>
#include <queue>

#include "xtensor/xarray.hpp"

#include "Datatypes.h"

class Node {
public:
    data_t threshold;
    unsigned int feature;
    std::vector<data_t> preds;

    Node(data_t threshold, unsigned int feature) : threshold(threshold), feature(feature) {}
    Node() = default;
};

template <bool RANDOM = true>
class Tree {
private:
    std::vector<Node> nodes;
    unsigned int start_leaf;
    unsigned int n_nodes;
    unsigned int n_classes;

    inline unsigned int node_index(xt::xarray<data_t> const &X, unsigned int const row) const {
        unsigned int idx = 0;

        while(idx < start_leaf) {
            auto const col = nodes[idx].feature;
            if (X(row, col) <= nodes[idx].threshold) {
                idx = 2*idx + 1;
            } else {
                idx = 2*idx + 2;
            }
        }
        return idx;
    }

    void random_nodes(unsigned long seed, xt::xarray<data_t> const &X, xt::xarray<unsigned int> const &Y) {
        std::mt19937 gen(seed);
        std::uniform_int_distribution<> idis(0, X.shape()[1] - 1);
        std::uniform_real_distribution<> fdis(0,1);
        //xt::xarray<data_t> amin = xt::amin(X, 0);
        //xt::xarray<data_t> amax = xt::amax(X, 0);

        nodes.resize(n_nodes);
        for (unsigned int i = 0; i < n_nodes; ++i) {
            auto feature =  idis(gen);
            //std::uniform_real_distribution<> fdis(amin(feature), amax(feature));
            nodes[i].feature = feature;
            nodes[i].threshold = fdis(gen);

            if (i >= start_leaf) {
                nodes[i].preds.resize(n_classes);
                std::fill(nodes[i].preds.begin(), nodes[i].preds.end(), 0);
            } 
        }

        for (unsigned int i = 0; i < X.shape()[0]; ++i) {
            nodes[node_index(X, i)].preds[Y(i)] += 1;
        }

        for (unsigned int i = start_leaf; i < n_nodes; ++i) {
            auto & preds = nodes[i].preds;
            data_t sum = std::accumulate(preds.begin(), preds.end(),data_t(0));
            if (sum > 0) {
                std::transform(preds.begin(), preds.end(), preds.begin(), [sum](auto& c){return 1.0/sum*c;});
            } else {
                std::fill(preds.begin(), preds.end(), 1.0/n_classes);
            }
        }
    }

    static data_t gini(std::vector<unsigned int> const &left, std::vector<unsigned int> const &right) {
        unsigned int sum_left = std::accumulate(left.begin(), left.end(), data_t(0));
        unsigned int sum_right = std::accumulate(right.begin(), right.end(), data_t(0));

        data_t p_left = 0;
        for (auto const l : left) {
            p_left += l / sum_left * l / sum_left;
        }

        data_t p_right = 0;
        for (auto const r : right) {
            p_right += r / sum_right * r / sum_right;
        }

        return sum_left / static_cast<data_t>(sum_left + sum_right) * p_left + sum_right /  static_cast<data_t>(sum_left + sum_right) * p_right;
    }

    static auto best_split(unsigned int n_classes, std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        if (X.size() <= 1) {
            return std::make_pair<data_t, unsigned int>(1.0, 0);
        }

        unsigned int n_data = X.size();
        unsigned int n_features = X[0].size();

        data_t overall_best_gini = 0;
        unsigned int overall_best_feature = 0;
        unsigned int overall_best_sample = 0;
        for (unsigned int i = 0; i < n_features; ++i) {
            // Usually we should use a view / column here to access columns but I have the feeling that this is slow?
            // TODO Verfiy
            std::vector<std::pair<data_t, unsigned int>> f_values(n_data);
            for (unsigned int j = 0; j < n_data; ++j) {
                f_values[j] = std::make_pair(X[j][i], Y[j]);
            }
            
            // By default sort sorts after the first feature
            std::sort(f_values.begin(), f_values.end());

            std::vector<unsigned int> left_cnts(n_classes);
            std::vector<unsigned int> right_cnts(n_classes);
            std::fill(left_cnts.begin(), left_cnts.end(), 0);
            std::fill(right_cnts.begin(), right_cnts.end(), 0);
            
            for (unsigned int j = 0; j < f_values.size(); ++j) {
                auto const & f = f_values[j];
                if (j == 0) {
                    left_cnts[f.second] += 1;
                } else {
                    right_cnts[f.second] += 1;
                }
            }
            
            data_t best_gini = gini(left_cnts, right_cnts);
            unsigned int best_sample = 0;

            for (unsigned int j = 1; j < f_values.size() - 1; ++j) {
                auto const & f = f_values[j];
                left_cnts[f.second] += 1;
                right_cnts[f.second] -= 1;
                data_t cur_gini = gini(left_cnts, right_cnts);
                
                if (cur_gini > best_gini) {
                    best_gini = cur_gini;
                    best_sample = j;
                }
            }

            if (i == 0 || best_gini > overall_best_gini) {
                overall_best_gini = best_gini;
                overall_best_feature = i;
                overall_best_sample = best_sample;
            } 
        }

        return std::make_pair(X[overall_best_sample][overall_best_feature], overall_best_feature);
    }

    void trained_nodes(xt::xarray<data_t> const &X, xt::xarray<unsigned int> const &Y) {
        // Note: Yeah this is kinda stupid to do I know. We have xtensor and xtensor has nice views and slices 
        // and all that good stuff. However, xtensor is sometimes slow when it comes to views and we have to 
        // handle indices which can be buggy sometimes. Moreover, stacking is also a pain sometimes. 
        // Plus, this code is actually kinda efficient.
        std::vector<std::vector<data_t>> XVec(X.shape()[0]);
        std::vector<unsigned int> YVec(Y.shape()[0]);

        for (unsigned int i = 0; i < X.shape()[0]; ++i) {
            XVec[i].resize(X.shape()[1]);
            for (unsigned int j = 0; j < X.shape()[1]; ++j) {
                XVec[i][j] = X(i,j);
            }
            YVec[i] = Y(i);
        }

        // <std::pair<std::vector<std::vector<data_t>>, std::vector<unsigned int>>
        std::queue<
            std::pair<std::vector<std::vector<data_t>>, std::vector<unsigned int>>
        > to_expand; // = {std::make_pair(XVec,YVec)};
        to_expand.push(std::make_pair(XVec, YVec));

        // auto split = best_split(n_classes, XVec, YVec);
        // nodes.push_back(Node(split.first, split.second));
        nodes.reserve(n_nodes);
        while(nodes.size() < start_leaf) {
            auto data = to_expand.front();
            to_expand.pop();

            auto split = best_split(n_classes, data.first, data.second);
            auto f = split.second;
            auto t = split.first;

            // We assume complete trees in this implementation which means that we always have 2 children 
            // and that each path in the tree has max_depth length. Now it might happen that XLeft / XRight is empty. 
            // In this best_split return t = 1, which means that _all_ data points are routed towards XLeft
            // and we keep on adding nodes as long as required to built the complete tree
            nodes.push_back(Node(t, f));
            
            std::vector<std::vector<data_t>> XLeft, XRight;
            std::vector<unsigned int> YLeft, YRight;
            for (unsigned int i = 0; i < data.first.size(); ++i) {
                if (data.first[i][f] <= t) {
                    XLeft.push_back(data.first[i]);
                    YLeft.push_back(data.second[i]);
                } else {
                    XRight.push_back(data.first[i]);
                    YRight.push_back(data.second[i]);
                }
            }

            to_expand.push(std::make_pair(XLeft, YLeft));
            to_expand.push(std::make_pair(XRight, YRight));
        }

        while(to_expand.size() > 0) {
            auto data = to_expand.front();
            to_expand.pop();
            Node n;
            n.preds.resize(n_classes);
            std::fill(n.preds.begin(), n.preds.end(), 0);

            data_t sum = data.second.size();
            for (auto const l : data.second) {
                n.preds[l] += 1;
            }

            if (sum > 0) {
                std::transform(n.preds.begin(), n.preds.end(), n.preds.begin(), [sum](auto& c){return 1.0/sum*c;});
            } else {
                std::fill(n.preds.begin(), n.preds.end(), 1.0/n_classes);
            }
            nodes.push_back(n);
        }
    }

public:

    Tree(unsigned int max_depth, unsigned int n_classes, unsigned long seed, xt::xarray<data_t> const &X, xt::xarray<unsigned int> const &Y) 
        : n_classes(n_classes) {

        start_leaf = std::pow(2,max_depth - 1) - 1;
        n_nodes = std::pow(2,max_depth) - 1;

        if constexpr(RANDOM) {
            random_nodes(seed, X, Y);
        } else {
            trained_nodes(X, Y);
        }
    }

    void predict_proba(xt::xarray<data_t> const &X, xt::xarray<data_t> &place_to_put, int row) {
        for (unsigned int i = 0; i < X.shape()[0]; ++i) {
            std::vector<data_t> const & xpred = nodes[node_index(X, i)].preds;
            for (unsigned int j = 0; j < xpred.size(); ++j) {
                place_to_put(row, i, j) = xpred[j];
            }
        }
    }

    xt::xarray<data_t> predict_proba(xt::xarray<data_t> const &X) {
        xt::xarray<data_t> preds;
        if (X.dimension() == 1) {
            preds = xt::xarray<data_t>::from_shape({1, n_classes});
        } else {
            preds = xt::xarray<data_t>::from_shape({X.size(), n_classes});
        }
        predict_proba(X, preds, 0);
        return preds;
    }
};

#endif