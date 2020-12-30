#ifndef BIASED_PROX_ENSEMBLE_H
#define BIASED_PROX_ENSEMBLE_H

#include <vector>
#include <math.h>
#include <random>
#include <algorithm>

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xadapt.hpp"

#include "Datatypes.h"
#include "Tree.h"
#include "Losses.h"

/**
 * @brief  The main reason why this interface exists, is because it makes class instansiation a little easier for the Pythonbindings. See Python.cpp for details.
 * @note   
 * @retval None
 */
template <typename pred_t>
class BiasedProxedEnsembleInterface {
public:
    virtual data_t next(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) = 0;

    virtual std::vector<std::vector<data_t>> predict_proba(std::vector<std::vector<data_t>> const &X) = 0;
    
    virtual std::vector<data_t> weights() const = 0;

    virtual unsigned int num_trees() const = 0;

    virtual ~BiasedProxedEnsembleInterface() { }
};

template <TREE_INIT tree_init, TREE_NEXT tree_next, typename pred_t>
class BiasedProxEnsemble : public BiasedProxedEnsembleInterface<pred_t> {

private:
    std::vector< Tree<tree_init, tree_next, pred_t> > _trees;
    std::vector<data_t> _weights;

    unsigned int max_depth;
    unsigned int max_trees;
    unsigned int n_classes;
    unsigned long seed;
    data_t step_size;
    data_t lambda;
    data_t const init_weight;
    std::vector<bool> const is_nominal;

    std::function< xt::xarray<data_t>(xt::xarray<data_t> const &, xt::xarray<data_t> const &) > loss;
    std::function< xt::xarray<data_t>(xt::xarray<data_t> const &, xt::xarray<data_t> const &) > loss_deriv;

public:

    BiasedProxEnsemble(
        unsigned int max_depth,
        unsigned int max_trees,
        unsigned int n_classes, 
        unsigned long seed, 
        data_t step_size,
        data_t lambda,
        data_t init_weight,
        std::vector<bool> const &is_nominal,
        std::function< xt::xarray<data_t>(xt::xarray<data_t> const &, xt::xarray<data_t> const &) > loss,
        std::function< xt::xarray<data_t>(xt::xarray<data_t> const &, xt::xarray<data_t> const &) > loss_deriv
    ) : max_depth(max_depth), max_trees(max_trees), n_classes(n_classes), seed(seed), step_size(step_size), lambda(lambda), init_weight(init_weight), is_nominal(is_nominal), loss(loss), loss_deriv(loss_deriv) {}

    data_t next(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        xt::xarray<unsigned int> y_tensor = xt::xarray<unsigned int>::from_shape({Y.size()});
        for (unsigned int i = 0; i < Y.size(); ++i) {
            y_tensor(i) = Y[i];
        }

        if (max_trees == 0 || _trees.size() < max_trees) {
            // Create new trees
            _weights.push_back(init_weight);
            _trees.push_back(Tree<tree_init, tree_next, pred_t>(max_depth, n_classes, seed++, X, Y, is_nominal));
        }

        xt::xarray<data_t> all_proba = xt::xarray<data_t>::from_shape({_trees.size(), X.size(), n_classes});
        for (unsigned int i = 0; i < _trees.size(); ++i) {
            _trees[i].predict_proba(X, all_proba, i);
        }
        auto w_tensor = xt::adapt(_weights, {(int)_weights.size()});
        auto all_proba_weighted = all_proba * xt::reshape_view(w_tensor, { (int)_weights.size(), 1, 1}); 
        
        // Compute the ensemble prediction as weighted sum.  
        xt::xarray<data_t> output = xt::mean(all_proba_weighted, 0);
        xt::xarray<data_t> losses = loss(output, y_tensor);
        xt::xarray<data_t> losses_deriv = loss_deriv(output, y_tensor);

        xt::xarray<data_t> dir_per_tree = all_proba * losses_deriv;
        for (unsigned int i = 0; i < _trees.size(); ++i) {
            _trees[i].next(X, Y, dir_per_tree, w_tensor(i) * step_size, i);
        }

        xt::xarray<data_t> directions = xt::mean(dir_per_tree, {1,2});
        w_tensor = w_tensor - step_size * directions;
        xt::xarray<data_t> sign = xt::sign(w_tensor);
        w_tensor = xt::abs(w_tensor) - lambda;
        w_tensor = sign*xt::maximum(w_tensor,0);

        auto wit = _weights.begin();
        auto tit = _trees.begin();

        while (wit != _weights.end() && tit != _trees.end()) {
            if (*wit == 0) {
                wit = _weights.erase(wit);
                tit = _trees.erase(tit);
            } else {
                ++wit;
                ++tit;
            }
        }

        return xt::sum(losses)();
    }

    std::vector<std::vector<data_t>> predict_proba(std::vector<std::vector<data_t>> const &X) {
        std::vector<std::vector<data_t>> output(X.size());
        if (_trees.size() == 0) {
            for (unsigned int i = 0; i < X.size(); ++i) {
                std::vector<data_t> tmp(n_classes);
                std::fill(tmp.begin(), tmp.end(), 1.0/n_classes);
                output[i] = tmp;
            }
        } else {
            xt::xarray<data_t> all_proba = xt::xarray<data_t>::from_shape({_trees.size(), X.size(), n_classes});
            for (unsigned int i = 0; i < _trees.size(); ++i) {
                _trees[i].predict_proba(X, all_proba, i);
            }
            auto w_tensor = xt::adapt(_weights, {(int)_weights.size()});
            auto all_proba_weighted = all_proba * xt::reshape_view(w_tensor, { (int)_weights.size(), 1, 1}); 
            
            xt::xarray<data_t> preds = xt::mean(all_proba_weighted, 0);
            for (unsigned int i = 0; i < X.size(); ++i) {
                output[i].resize(n_classes);
                for (unsigned int j = 0; j < n_classes; ++j) {
                    output[i][j] = preds(i,j);
                }
            }
        }
        return output;
    }

    unsigned int num_trees() const {
        return _trees.size();
    }

    std::vector<data_t> weights() const {
        return _weights;
    }
};

#endif