#ifndef BIASED_PROX_ENSEMBLE_H
#define BIASED_PROX_ENSEMBLE_H

#include <vector>
#include <math.h>
#include <random>
#include <algorithm>

#include "Datatypes.h"
#include "Tree.h"
#include "Losses.h"

class BiasedProxEnsemble {

private:
    std::vector<Tree> trees;
    std::vector<data_t> weights;

    unsigned int max_depth;
    unsigned int n_classes;
    unsigned long seed;
    data_t alpha;
    data_t lambda;

    // maybe expose that to somewhere
    data_t const init_weight = 0.0;
public:

    BiasedProxEnsemble(
        unsigned int max_depth,
        unsigned int n_classes, 
        unsigned long seed, 
        data_t alpha,
        data_t lambda
    ) : max_depth(max_depth), n_classes(n_classes), seed(seed), alpha(alpha), lambda(lambda) {}

    data_t next(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        // Create new trees
        weights.push_back(init_weight);
        trees.push_back(Tree(max_depth, n_classes, seed++, X, Y));

        // Perform predictions of all trees on current batch
        std::vector<std::vector<std::vector<data_t>>> all_proba(trees.size());

        for (unsigned int i = 0; i < trees.size(); ++i) {
            all_proba[i] = trees[i].predict_proba(X);
        }

        // Compute the ensemble prediction as weighted sum.  
        std::vector<std::vector<data_t>> output(X.size());
        for (unsigned int j = 0; j < trees.size(); ++j) {
            for (unsigned int i = 0; i < X.size(); ++i) {
                if (output[i].size() < n_classes) {
                    output[i].resize(n_classes);
                    std::fill(output[i].begin(), output[i].end(), 0);
                }
                for (unsigned int c = 0; c < n_classes; ++c) {
                    output[i][c] += weights[j] * all_proba[j][i][c];
                }
            }
        }
        
        // Compute the losses
        std::vector<std::vector<data_t>> loss(X.size());
        std::vector<std::vector<data_t>> loss_deriv(X.size());

        for (unsigned int i = 0; i < X.size(); ++i) {
            loss[i] = cross_entropy(output[i], Y[i]);
            loss_deriv[i] = cross_entropy_deriv(output[i], Y[i]);
        }

        // Perform the prox-sgd step with biased gradient
        std::vector<data_t> new_weights;
        std::vector<Tree> new_trees;

        for (unsigned int i = 0; i < trees.size(); ++i) {

            // Compute the direction
            data_t dir = 0;
            for (unsigned int j = 0; j < X.size(); ++j) {
                dir += std::inner_product(all_proba[i][j].begin(), all_proba[i][j].end(), loss_deriv[j].begin(), data_t(0));
            }
            dir = dir / (X.size() * n_classes);

            // Perform the prox
            auto tmp_w = weights[i] - alpha * dir;
            auto sign = (data_t(0) < tmp_w) - (tmp_w < data_t(0));
            tmp_w = std::abs(tmp_w) - lambda;
            weights[i] = sign * std::max(tmp_w, 0.0);

            // Only store this tree if it has a nonzero weight
            if (weights[i] != 0) {
                new_weights.push_back(weights[i]);
                new_trees.push_back(trees[i]);
            }
        }

        weights = new_weights;
        trees = new_trees;

        // Return the average loss over all classes / examples in the batch
        data_t l = 0;
        for (auto const & iloss: loss) {
            l += std::accumulate(iloss.begin(), iloss.end(), data_t(0));
        }
        return l; // / (loss.size() * n_classes);
    }

    std::vector<std::vector<data_t>> predict_proba(std::vector<std::vector<data_t>> const &X) {
        // Perform predictions of all trees on current batch
        std::vector<std::vector<std::vector<data_t>>> all_proba;
        all_proba.reserve(trees.size());

        for (unsigned int i = 0; i < trees.size(); ++i) {
            all_proba[i] = trees[i].predict_proba(X);
        }

        // Compute the ensemble prediction as weighted sum.  
        std::vector<std::vector<data_t>> output;
        output.reserve(X.size());
        for (unsigned int i = 0; i < X.size(); ++i) {
            output[i].reserve(n_classes);
            std::fill(output[i].begin(), output[i].end(), 0);

            for (unsigned int j = 0; j < trees.size(); ++j) {
                for (unsigned int c = 0; c < n_classes; ++c) {
                    output[i][c] += weights[j] * all_proba[i][j][c];
                }
            }
        }

        return output;
    }

    std::vector<data_t> predict_proba(std::vector<data_t> const &x) {
        return predict_proba({x});
    }

    unsigned int num_trees() const {
        return trees.size();
    }

};

#endif