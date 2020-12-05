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

class Tree {
private:
    std::vector<Node> nodes;
    unsigned int start_leaf;
    unsigned int n_nodes;
    unsigned int n_classes;

    inline unsigned int node_index(std::vector<data_t> const &x) const {
        unsigned int idx = 0;

        while(idx < start_leaf) {
            auto const f = nodes[idx].feature;
            if (x[f] <= nodes[idx].threshold) {
                idx = 2*idx + 1;
            } else {
                idx = 2*idx + 2;
            }
        }
        return idx;
    }

    void random_nodes(unsigned long seed, std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        std::mt19937 gen(seed);
        std::uniform_int_distribution<> idis(0, X[0].size() - 1);
        std::uniform_real_distribution<> fdis(0,1);

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

        for (unsigned int i = 0; i < X.size(); ++i) {
            nodes[node_index(X[i])].preds[Y[i]] += 1;
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

        data_t gleft = 0;
        for (auto const l : left) {
            gleft += (static_cast<data_t>(l) / sum_left) * (static_cast<data_t>(l) / sum_left);
        }
        gleft = 1.0 - gleft;

        data_t gright = 0;
        for (auto const r : right) {
            gright += (static_cast<data_t>(r) / sum_right) * (static_cast<data_t>(r) / sum_right);
        }
        gright = 1.0 - gright;

        return sum_left / static_cast<data_t>(sum_left + sum_right) * gleft + sum_right /  static_cast<data_t>(sum_left + sum_right) * gright;
    }

    // TODO CHECK IF THIS REALLY BUILTS A "PERFECT" TREE
    static auto best_split(unsigned int n_classes, std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        if (X.size() <= 1) {
            return std::make_pair<data_t, unsigned int>(1.0, 0);
        }

        // if (X.size() == 2) {
        //     return std::make_pair<data_t, unsigned int>(static_cast<data_t>(X[0][0]), 0);
        // }

        unsigned int n_data = X.size();
        unsigned int n_features = X[0].size();

        data_t overall_best_gini = 0;
        unsigned int overall_best_feature = 0;
        data_t overall_best_threshold = 0;
        for (unsigned int i = 0; i < n_features; ++i) {
            std::vector<std::pair<data_t, unsigned int>> f_values(n_data);
            for (unsigned int j = 0; j < n_data; ++j) {
                f_values[j] = std::make_pair(X[j][i], Y[j]);
            }
            
            // By default sort sorts after the first feature
            std::sort(f_values.begin(), f_values.end());
            data_t max_t = f_values[f_values.size() - 1].first;

            std::vector<unsigned int> left_cnts(n_classes);
            std::vector<unsigned int> right_cnts(n_classes);
            std::fill(left_cnts.begin(), left_cnts.end(), 0);
            std::fill(right_cnts.begin(), right_cnts.end(), 0);
            
            left_cnts[f_values[0].second] += 1;
            for (unsigned int j = 1; j < f_values.size(); ++j) {
                auto const & f = f_values[j];
                right_cnts[f.second] += 1;
            }
            
            data_t best_gini = gini(left_cnts, right_cnts);
            data_t best_threshold = 0.5 * (f_values[0].first + f_values[1].first); 
            // std::cout << "Checking feature " << 0 << " with threshold " << 0.5 * (f_values[0].first + f_values[1].first) << " and score " << best_gini << std::endl;

            // for (unsigned int j = 1; j < f_values.size() - 1; ++j) {
            unsigned int j = 1;
            while (f_values[j].first < max_t) {
                auto const & f = f_values[j];
                left_cnts[f.second] += 1;
                right_cnts[f.second] -= 1;

                if (f_values[j - 1].first != f_values[j].first) {
                    data_t cur_gini = gini(left_cnts, right_cnts);
                    // std::cout << "Checking feature " << i << " with threshold " << 0.5 * (f_values[j].first + f_values[j + 1].first) << " and score " << cur_gini << std::endl;
                    if (cur_gini < best_gini) {
                        best_gini = cur_gini;
                        best_threshold = 0.5 * (f_values[j].first + f_values[j + 1].first);
                    }
                }
                ++j;
            }

            // TODO ADD RANDOM STUFF IN CASE OF A TIE
            if (i == 0 || best_gini < overall_best_gini) {
                overall_best_gini = best_gini;
                overall_best_feature = i;
                overall_best_threshold = best_threshold;
            } 
        }

        // std::cout << "Best split is " << overall_best_feature << " with threshold " << overall_best_threshold <<  std::endl;
        return std::make_pair(overall_best_threshold, overall_best_feature);
    }

    void trained_nodes(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        // <std::pair<std::vector<std::vector<data_t>>, std::vector<unsigned int>>
        std::queue<
            std::pair<std::vector<std::vector<data_t>>, std::vector<unsigned int>>
        > to_expand; 
        to_expand.push(std::make_pair(X, Y));

        // auto split = best_split(n_classes, XVec, YVec);
        // nodes.push_back(Node(split.first, split.second));
        nodes.reserve(n_nodes);
        while(nodes.size() < start_leaf) {
            auto data = to_expand.front();
            to_expand.pop();

            auto split = best_split(n_classes, data.first, data.second);
            auto t = split.first;
            auto f = split.second;
            
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

            // std::cout << nodes.size() << " : Choose feature " << f << " with threshold " << t << std::endl;
            // std::cout << nodes.size() << " : Left has " << XLeft.size() << std::endl;
            // std::cout << nodes.size() << " : Right has " << XRight.size() << std::endl;
            // std::cout << std::endl;

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

    Tree(bool use_random, unsigned int max_depth, unsigned int n_classes, unsigned long seed, std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) : n_classes(n_classes) {

        start_leaf = std::pow(2,max_depth) - 1;
        n_nodes = std::pow(2,max_depth + 1) - 1;

        if (use_random) {
            random_nodes(seed, X, Y);
        } else {
            trained_nodes(X, Y);
        }
    }

    void predict_proba(std::vector<std::vector<data_t>> const &X, xt::xarray<data_t> &place_to_put, int row) {
        for (unsigned int i = 0; i < X.size(); ++i) {
            std::vector<data_t> const & xpred = nodes[node_index(X[i])].preds;
            for (unsigned int j = 0; j < xpred.size(); ++j) {
                place_to_put(row, i, j) = xpred[j];
            }
        }
    }

    std::vector<std::vector<data_t>>  predict_proba(std::vector<std::vector<data_t>> const &X) {
        std::vector<std::vector<data_t>> preds(X.size());
        for (unsigned int i = 0; i < X.size(); ++i) {
            std::vector<data_t> const & xpred = nodes[node_index(X[i])].preds;
            preds.push_back(xpred);
        }
        return preds;
    }
};

#endif