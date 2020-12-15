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

class BiasedProxEnsemble {

private:
    std::vector<Tree> _trees;
    std::vector<data_t> _weights;

    unsigned int max_depth;
    unsigned int max_trees;
    unsigned int n_classes;
    unsigned long seed;
    data_t step_size;
    data_t lambda;
    data_t const init_weight;
    TREE_INIT const tree_init;
    TREE_NEXT const tree_next;

    std::function< xt::xarray<data_t>(xt::xarray<data_t> const &, xt::xarray<data_t> const &) > loss;
    std::function< xt::xarray<data_t>(xt::xarray<data_t> const &, xt::xarray<data_t> const &) > loss_deriv;

public:

    // BiasedProxEnsemble(
    //     unsigned int max_depth,
    //     unsigned int n_classes, 
    //     unsigned long seed, 
    //     data_t alpha,
    //     data_t lambda,
    //     data_t init_weight = 0.0
    // ) : max_depth(max_depth), n_classes(n_classes), seed(seed), alpha(alpha), lambda(lambda), init_weight(init_weight), loss(mse), loss_deriv(mse_deriv) {}

    BiasedProxEnsemble(
        unsigned int max_depth,
        unsigned int max_trees,
        unsigned int n_classes, 
        unsigned long seed, 
        data_t step_size,
        data_t lambda,
        data_t init_weight,
        TREE_INIT tree_init,
        TREE_NEXT tree_next,
        std::function< xt::xarray<data_t>(xt::xarray<data_t> const &, xt::xarray<data_t> const &) > loss,
        std::function< xt::xarray<data_t>(xt::xarray<data_t> const &, xt::xarray<data_t> const &) > loss_deriv
    ) : max_depth(max_depth), max_trees(max_trees), n_classes(n_classes), seed(seed), step_size(step_size), lambda(lambda), init_weight(init_weight), tree_init(tree_init), tree_next(tree_next), loss(loss), loss_deriv(loss_deriv) {}

    data_t next(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        xt::xarray<unsigned int> y_tensor = xt::xarray<unsigned int>::from_shape({Y.size()});
        for (unsigned int i = 0; i < Y.size(); ++i) {
            y_tensor(i) = Y[i];
        }

        if (max_trees == 0 || _trees.size() < max_trees) {
            // Create new trees
            _weights.push_back(init_weight);

            _trees.push_back(Tree(tree_init, tree_next, max_depth, n_classes, seed++, X, Y));
        }

        xt::xarray<data_t> all_proba = xt::xarray<data_t>::from_shape({_trees.size(), X.size(), n_classes});
        for (unsigned int i = 0; i < _trees.size(); ++i) {
            _trees[i].predict_proba(X, all_proba, i);
        }
        // TODO CALL PREDICT_PROBA HERE?
        auto w_tensor = xt::adapt(_weights, {(int)_weights.size()});
        auto all_proba_weighted = all_proba * xt::reshape_view(w_tensor, { (int)_weights.size(), 1, 1}); 
        
        // Compute the ensemble prediction as weighted sum.  
        xt::xarray<data_t> output = xt::mean(all_proba_weighted, 0);
        // auto pred = xt::argmax(output, 1);
        // std::cout << "pred proba:" << output << std::endl;
        // std::cout << "w_tensor:" << w_tensor << std::endl;
        // std::cout << "pred:" << xt::argmax(output, 1) << std::endl;
        // std::cout << "target:" << y_tensor << std::endl;
        // data_t acc = 0;
        // for (unsigned int i = 0; i < y_tensor.shape()[0]; ++i) {
        //     if (y_tensor(i) == pred(i)) {
        //         acc++;
        //     }
        // }


        // WHY DOES THIS NEED LARGE STEPSIZES AROUND 0.1 - 0.5?
        // std::cout << "acc:" << (data_t)acc / y_tensor.shape()[0] << std::endl;
        // Compute the losses
        xt::xarray<data_t> losses = loss(output, y_tensor);
        xt::xarray<data_t> losses_deriv = loss_deriv(output, y_tensor);

        // std::cout << "LOSS: " << losses << std::endl;
        // std::cout << "LOSS DERIV: " << losses_deriv << std::endl;
        xt::xarray<data_t> dir_per_tree = all_proba * losses_deriv;
        // std::cout << "LOSS dir_per_tree: " << dir_per_tree << std::endl;
        for (unsigned int i = 0; i < _trees.size(); ++i) {
            _trees[i].next(X, Y, dir_per_tree, w_tensor(i) * step_size, i);
        }
        // std::cout << xt::adapt(dir_per_tree.shape()) << std::endl;
        xt::xarray<data_t> directions = xt::mean(dir_per_tree, {1,2});
        // std::cout << "1: " << w_tensor << std::endl;
        w_tensor = w_tensor - step_size * directions;
        // std::cout << "2: " << w_tensor << std::endl;
        xt::xarray<data_t> sign = xt::sign(w_tensor);
        w_tensor = xt::abs(w_tensor) - lambda;
        // std::cout << "3: " << w_tensor << std::endl;
        w_tensor = sign*xt::maximum(w_tensor,0);
        // std::cout << "4: " << w_tensor << std::endl;
        // std::cout << std::endl;

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

        // auto tmp_w = xt::adapt(weights, {(int)weights.size()});
        // auto tmp_weighted = all_proba * xt::reshape_view(tmp_w, { (int)weights.size(), 1, 1}); 
        // xt::xarray<data_t> tmp_output = xt::mean(tmp_weighted, 0);

        // std::cout << "all_proba:" << all_proba << std::endl;
        // std::cout << "tmp_output:" << tmp_output << std::endl;
        // std::cout << "output:" << output << std::endl;
        // xt::xarray<data_t> tmp_losses = loss(tmp_output, y_tensor);

        // auto before = xt::sum(losses)();
        // auto after = xt::sum(tmp_losses)();
        // std::cout << "dir:" << directions << std::endl;
        // std::cout << "proba:" << all_proba << std::endl;
        // std::cout << "tmp_losses:" << tmp_losses << std::endl;
        // std::cout << "losses:" << losses << std::endl;
        // std::cout << w_tensor << std::endl;
        // std::cout << "BEFORE UPDATE: " << before << " AFTER UPDATE: " <<  after << std::endl;
    
        // Return the sum of all loss in the batch
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