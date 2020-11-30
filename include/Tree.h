#ifndef TREE_H
#define TREE_H

#include <vector>
#include <math.h>
#include <random>
#include <algorithm>

#include "Datatypes.h"

class Node {
public:
    data_t threshold;
    unsigned int feature;
    // unsigned int left;
    // unsigned int right;
    // bool is_leaf;
    std::vector<data_t> preds;
};

class Tree {

private:
    std::vector<Node> nodes;
    unsigned int start_leaf;
    unsigned int n_nodes;

    inline unsigned int node_index(std::vector<data_t> const &x) const {
        unsigned int idx = 0;

         while(idx < start_leaf) {
            if (x[nodes[idx].feature] <= nodes[idx].threshold) {
                idx = 2*idx + 1;
            } else {
                idx = 2*idx + 2;
            }
        }

        return idx;
    }

public:

    Tree(unsigned int max_depth, unsigned int n_classes, unsigned long seed, std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        start_leaf = std::pow(2,max_depth - 1) - 1;
        n_nodes = std::pow(2,max_depth) - 1;
        nodes.resize(n_nodes);

        std::mt19937 gen(seed);
        std::uniform_real_distribution<> fdis(0, 1);
        std::uniform_int_distribution<> idis(0, X[0].size() - 1);

        for (unsigned int i = 0; i < n_nodes; ++i) {
            nodes[i].threshold = fdis(gen);
            nodes[i].feature = idis(gen);
            if (i >= start_leaf) {
                // nodes[i].is_leaf = true;
                nodes[i].preds.resize(n_classes);
                std::fill(nodes[i].preds.begin(), nodes[i].preds.end(), 0);
            } 
            // else {
            //     nodes[i].is_leaf = false;
            // }
        }

        for (unsigned int i = 0; i < X.size(); ++i) {
            auto const & x = X[i];
            auto const & y = Y[i];
            nodes[node_index(x)].preds[y] = 1;
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

    std::vector<data_t> predict_proba(std::vector<data_t> const &x) {
        return nodes[node_index(x)].preds;
    }

    std::vector<std::vector<data_t>> predict_proba(std::vector<std::vector<data_t>> const &X) {
        std::vector<std::vector<data_t>> preds(X.size());
        
        // Apparently assignment is quicker than push_back
        // https://www.acodersjourney.com/6-tips-supercharge-cpp-11-vector-performance/
        for (unsigned int i = 0; i < X.size(); ++i) {
            preds[i] = predict_proba(X[i]);
        }
        return preds;
    }

};

#endif