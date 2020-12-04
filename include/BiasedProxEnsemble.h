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

template <bool RANDOM = true>
class BiasedProxEnsemble {

private:
    std::vector<Tree<RANDOM>> trees;
    std::vector<data_t> weights;

    unsigned int max_depth;
    unsigned int n_classes;
    unsigned long seed;
    data_t alpha;
    data_t lambda;
    data_t const init_weight;

    std::function< xt::xarray<data_t>(xt::xarray<data_t> const &, xt::xarray<data_t> const &) > loss;
    std::function< xt::xarray<data_t>(xt::xarray<data_t> const &, xt::xarray<data_t> const &) > loss_deriv;

public:

    BiasedProxEnsemble(
        unsigned int max_depth,
        unsigned int n_classes, 
        unsigned long seed, 
        data_t alpha,
        data_t lambda,
        data_t init_weight = 0.0
    ) : max_depth(max_depth), n_classes(n_classes), seed(seed), alpha(alpha), lambda(lambda), init_weight(init_weight), loss(mse), loss_deriv(mse_deriv) {}

    BiasedProxEnsemble(
        unsigned int max_depth,
        unsigned int n_classes, 
        unsigned long seed, 
        data_t alpha,
        data_t lambda,
        std::function< xt::xarray<data_t>(xt::xarray<data_t> const &, xt::xarray<data_t> const &) > loss,
        std::function< xt::xarray<data_t>(xt::xarray<data_t> const &, xt::xarray<data_t> const &) > loss_deriv,
        data_t init_weight = 0.0
    ) : max_depth(max_depth), n_classes(n_classes), seed(seed), alpha(alpha), lambda(lambda), init_weight(init_weight), loss(loss), loss_deriv(loss_deriv) {}

    data_t next(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        xt::xarray<unsigned int> y_tensor = xt::xarray<unsigned int>::from_shape({Y.size()});
        for (unsigned int i = 0; i < Y.size(); ++i) {
            y_tensor(i) = Y[i];
        }

        // Create new trees
        weights.push_back(init_weight);

        //trees.push_back(Tree(max_depth, n_classes, seed++, x_tensor, y_tensor));
        trees.push_back(Tree<RANDOM>(max_depth, n_classes, seed++, X, Y));

        xt::xarray<data_t> all_proba = xt::xarray<data_t>::from_shape({trees.size(), X.size(), n_classes});
        //Tensor<data_t> all_proba(trees.size(), X.size(), n_classes);
        for (unsigned int i = 0; i < trees.size(); ++i) {
            trees[i].predict_proba(X, all_proba, i);
        }
        // TODO CALL PREDICT_PROBA HERE?
        auto w_tensor = xt::adapt(weights, {(int)weights.size()});
        auto all_proba_weighted = all_proba * xt::reshape_view(w_tensor, { (int)weights.size(), 1, 1}); 
        
        // Compute the ensemble prediction as weighted sum.  
        xt::xarray<data_t> output = xt::mean(all_proba_weighted, 0);
        
        // Compute the losses
        xt::xarray<data_t> losses = loss(output, y_tensor);
        xt::xarray<data_t> losses_deriv = loss_deriv(output, y_tensor);
        // std::cout << losses << std::endl;
        // std::cout << losses_deriv << std::endl;

        //auto tmp = all_proba * losses_deriv;
        xt::xarray<data_t> directions = xt::mean(all_proba * losses_deriv, {1,2});
        w_tensor = w_tensor - alpha * directions;
        // std::cout << "dir " << directions << std::endl;
        // std::cout << "all_proba " << all_proba << std::endl;
        // std::cout << "wtensor " << w_tensor << std::endl;
        // std::cout << "tmp " << tmp << std::endl;
        xt::xarray<data_t> sign = xt::sign(w_tensor);
        w_tensor = xt::abs(w_tensor) - lambda;
        w_tensor = sign*xt::maximum(w_tensor,0);

        auto wit = weights.begin();
        auto tit = trees.begin();

        while (wit != weights.end() && tit != trees.end()) {
            if (*wit == 0) {
                wit = weights.erase(wit);
                tit = trees.erase(tit);
            } else {
                ++wit;
                ++tit;
            }
        }

        // Return the sum of all loss in the batch
        return xt::sum(losses)();
    }

    std::vector<std::vector<data_t>> predict_proba(std::vector<std::vector<data_t>> const &X) {
        std::vector<std::vector<data_t>> output(X.size());
        if (trees.size() == 0) {
            for (unsigned int i = 0; i < X.size(); ++i) {
                std::vector<data_t> tmp(n_classes);
                std::fill(tmp.begin(), tmp.end(), 1.0/n_classes);
                output[i] = tmp;
            }
            // xt::xarray<data_t> output;
            // output = xt::xarray<data_t>::from_shape({1, X.size(), n_classes});
            // output.fill(1.0/n_classes);
        } else {
            xt::xarray<data_t> all_proba = xt::xarray<data_t>::from_shape({trees.size(), X.size(), n_classes});
            for (unsigned int i = 0; i < trees.size(); ++i) {
                trees[i].predict_proba(X, all_proba, i);
            }
            auto w_tensor = xt::adapt(weights, {(int)weights.size()});
            auto all_proba_weighted = all_proba * xt::reshape_view(w_tensor, { (int)weights.size(), 1, 1}); 
            
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
        return trees.size();
    }

};

#endif