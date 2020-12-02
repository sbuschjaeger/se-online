#ifndef TREE_H
#define TREE_H

#include <vector>
#include <math.h>
#include <random>
#include <algorithm>

#include "xtensor/xarray.hpp"

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
    unsigned int n_classes;

    inline unsigned int node_index(xt::xarray<data_t> const &X, unsigned int const row) const {
    //inline unsigned int node_index(std::vector<data_t> const &x) const {
        unsigned int idx = 0;

        while(idx < start_leaf) {
            auto const col = nodes[idx].feature;
            //if (x(nodes[idx].feature) <= nodes[idx].threshold) {
            if (X(row, col) <= nodes[idx].threshold) {
                idx = 2*idx + 1;
            } else {
                idx = 2*idx + 2;
            }
        }

        return idx;
    }

public:

    Tree(unsigned int max_depth, unsigned int n_classes, unsigned long seed, xt::xarray<data_t> const &X, xt::xarray<unsigned int> const &Y) 
    //Tree(unsigned int max_depth, unsigned int n_classes, unsigned long seed, std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) 
        : n_classes(n_classes) {

        start_leaf = std::pow(2,max_depth - 1) - 1;
        n_nodes = std::pow(2,max_depth) - 1;
        nodes.resize(n_nodes);

        std::mt19937 gen(seed);
        std::uniform_real_distribution<> fdis(0, 1);
        std::uniform_int_distribution<> idis(0, X.shape()[1] - 1);
        //std::uniform_int_distribution<> idis(0, X[0].size() - 1);

        for (unsigned int i = 0; i < n_nodes; ++i) {
            nodes[i].threshold = fdis(gen);
            nodes[i].feature = idis(gen);
            if (i >= start_leaf) {
                nodes[i].preds.resize(n_classes);
                std::fill(nodes[i].preds.begin(), nodes[i].preds.end(), 0);
            } 
        }

        for (unsigned int i = 0; i < X.shape()[0]; ++i) {
        //for (unsigned int i = 0; i < X.size(); ++i) {
            //auto && x = xt::row(X, i);
            nodes[node_index(X, i)].preds[Y[i]] = 1;
            //nodes[node_index(X[i])].preds[Y[i]] = 1;
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

    //void predict_proba(std::vector<std::vector<data_t>> const &X, xt::xarray<data_t> &place_to_put, int row) {
    void predict_proba(xt::xarray<data_t> const &X, xt::xarray<data_t> &place_to_put, int row) {
        //for (unsigned int i = 0; i < X.size(); ++i) {
        for (unsigned int i = 0; i < X.shape()[0]; ++i) {
            //auto && x = xt::row(X, i);
            //auto x = xt::view(X, i, xt::all());
            //std::vector<data_t> const & x = X[i];
            std::vector<data_t> const & xpred = nodes[node_index(X, i)].preds;
            for (unsigned int j = 0; j < xpred.size(); ++j) {
                place_to_put(row, i, j) = xpred[j];
            }
        }
    }

    xt::xarray<data_t> predict_proba(xt::xarray<data_t> const &X) {
        // TODO TEST THIS FUNCTION
        xt::xarray<data_t> preds;
        if (X.dimension() == 1) {
            preds = xt::xarray<data_t>::from_shape({1, n_classes});
        } else {
            preds = xt::xarray<data_t>::from_shape({X.size(), n_classes});
        }
        predict_proba(X, preds, 0);
        return preds;
    }

    // void predict_proba(xt::xarray<data_t> const &X, xt::xarray<data_t> &place_to_put, int row) {
    //     for (unsigned int i = 0; i < X.shape()[0]; ++i) {
    //         auto x = xt::view(X, i, xt::all());
    //         std::vector<data_t> const & xpred = nodes[node_index(x)].preds;
    //         for (unsigned int j = 0; j < xpred.size(); ++j) {
    //             place_to_put(row, i, j) = xpred[j];
    //         }
    //     }
    // }

};

#endif