#ifndef TREE_H
#define TREE_H

#include <vector>
#include <math.h>
#include <random>
#include <algorithm>
#include <queue>

#include "xtensor/xarray.hpp"

#include "Datatypes.h"

enum TREE_INIT {TRAIN, FULLY_RANDOM, RANDOM};
enum TREE_NEXT {GRADIENT, NONE, INCREMENTAL};

class Node {
public:
    data_t threshold;
    unsigned int feature;
    unsigned int total_cnt;
    std::vector<data_t> preds;

    Node(data_t threshold, unsigned int feature) : threshold(threshold), feature(feature), total_cnt(0) {}
    Node() = default;
};

class Tree {
private:
    std::vector<Node> nodes;
    unsigned int start_leaf;
    unsigned int n_nodes;
    unsigned int n_classes;
    unsigned long seed;
    TREE_NEXT tree_next;

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

    /**
     * @brief  Compute the weighted gini score for the given split. Weighted means here, that we weight the individual gini scores of left and right with the proportion of data in each child node. This leads to slightly more balanced splits.
     * @note   
     * @param  &left: Class-counts for the left child
     * @param  &right: Class-couts for the right child.
     * @retval The weighted gini score.
     */
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
    
    /**
     * @brief  Compute the best split for the given data. This algorithm has O(d * N log N) runtime, where N is the number of examples and d is the number of features.
     * This implementation ensures that the returned split splits the data so that each child is non-empty if applied to the given data (at-least one example form X is routed towards left and towards right). The only exception occurs if X is empty or contains one example. In this case we return feature 0 with threshold 1. Threshold-values are placed in the middle between two samples. 
     * If two splits are equally good, then the first split is chosen. Note that this introduces a slight bias towards the first features. 
     * TODO: This code assumes that all features are [0,1] for the X.size() <= 1. Change that 
     * TODO: Change code for tie-breaking
     * @note   
     * @param  &X: The example-set which is used to compute the splitting
     * @param  &Y: The label-set which is used to compute the splitting
     * @param  n_classes: The number of classes
     * @retval The best split as a std::pair<data_t, unsigned int>(best_threshold, best_feature) where the first entry is the threshold and the second entry the feature index.
     */
    static auto best_split(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned long n_classes) {
        if (X.size() <= 1) {
            return std::make_pair<data_t, unsigned int>(1.0, 0);
        }

        unsigned int n_data = X.size();
        unsigned int n_features = X[0].size();

        data_t overall_best_gini = 0;
        unsigned int overall_best_feature = 0;
        data_t overall_best_threshold = 0;
        bool split_set = false;
        for (unsigned int i = 0; i < n_features; ++i) {
            // In order to compute the best spliting threshold for the current feature we need to evaluate every possible split value.
            // These can be up to n_data - 1 points and for each threshold we need to evaluate if they belong to the left or right child. 
            // The naive implementation thus require O(n_data**2) runtime. We use a slightly more optimized version which requires O(n_data * log n_data). 
            // To do so, we first the examples according to their feature values and compute the initial statistics for the left/right child. Then, we gradually 
            // move the split-threshold to the next value and onyl update the statistics.

            // Copy feature values and targets into new vector
            std::vector<std::pair<data_t, unsigned int>> f_values(n_data);
            for (unsigned int j = 0; j < n_data; ++j) {
                f_values[j] = std::make_pair(X[j][i], Y[j]);
            }
            
            // By default sort sorts after the first feature
            std::sort(f_values.begin(), f_values.end());
            data_t max_t = f_values[f_values.size() - 1].first;

            // Prepare class statistics
            std::vector<unsigned int> left_cnts(n_classes);
            std::vector<unsigned int> right_cnts(n_classes);
            std::fill(left_cnts.begin(), left_cnts.end(), 0);
            std::fill(right_cnts.begin(), right_cnts.end(), 0);
            
            // Compute initial statistics if we split at the first possible split location. Usually we assume that
            // 0.5 * (f_values[0] + f_values[1]) would be an appropriate split value. However, in some edge cases
            // f_values[0] == f_values[1]. Thus, we first have to look for the first index j where f_values[0] != f_values[i]
            // for splitting. While doing so, we update the class counts for the left / right child.
            bool first = true;
            unsigned int begin = 0; 
            data_t best_threshold;
            for (unsigned int j = 0; j < f_values.size(); ++j) {
                if (f_values[j].first == f_values[0].first) {
                    left_cnts[f_values[j].second] += 1;
                } else {
                    if (first) {
                        best_threshold = 0.5 * (f_values[0].first + f_values[j].first); 
                        first = false;
                        begin++;
                    }
                    right_cnts[f_values[j].second] += 1;
                }
            }
            
            if (first) {
                // We never choose a threshold which means that f_values[0] = f_values[1] = .. = f_values[end]. 
                // This will not give us a good simplit, so ignore this feature
                break;
            }
            // Compute the corresponding gini score 
            data_t best_gini = gini(left_cnts, right_cnts);

            // Repeat what we have done above with the initial scanning, but now update left_cnts / right_cnts appropriately.
            unsigned int j = begin;
            while (f_values[j].first < max_t) {
                // Update the class statistics by virtually placing the current split threshold over the next example
                auto const & f = f_values[j];
                left_cnts[f.second] += 1;
                right_cnts[f.second] -= 1;

                // If some examples have the same feature, just ignore this. Only evaluate new splits where the feature value changes.
                if (f_values[j - 1].first != f_values[j].first) {
                    data_t cur_gini = gini(left_cnts, right_cnts);
                    if (cur_gini < best_gini) {
                        best_gini = cur_gini;
                        best_threshold = 0.5 * (f_values[j].first + f_values[j + 1].first);
                    }
                }
                ++j;
            }

            // Check if we not have already select a split or if this split is better than the other splits we found so far.
            // If so, then set this split
            if (!split_set || best_gini < overall_best_gini) {
                overall_best_gini = best_gini;
                overall_best_feature = i;
                overall_best_threshold = best_threshold;
                split_set = true;
            } 
        }

        return std::make_pair(overall_best_threshold, overall_best_feature);
    }

     /**
     * @brief  Compute a random split for the given data. This algorithm has O(d * log d + d * N) runtime in the worst case, but should usually run in O(d * log d + N), where N is the number of examples and d is the number of features.
     * This implementation ensures that the returned split splits the data so that each child is non-empty if applied to the given data (at-least one example form X is routed towards left and towards right). The only exception occurs if X is empty or contains one example. In this case we return feature 0 with threshold 1. Threshold-values are placed in the middle between two samples. 
     * TODO: This code assumes that all features are [0,1] for the X.size() <= 1 or in case of invalid splits. Change that 
     * @note   
     * @param  &X: The example-set which is used to compute the splitting
     * @param  &Y: The label-set which is used to compute the splitting
     * @param  n_classes: The number of classes
     * @retval The best split as a std::pair<data_t, unsigned int>(best_threshold, best_feature) where the first entry is the threshold and the second entry the feature index.
     */
    static auto random_split(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned long seed) {
        if (X.size() <= 1) {
            return std::make_pair<data_t, unsigned int>(1.0, 0);
        }

        // We want to split at a random feature. However, we also want to ensure that the left / right child receive at-least one example with this random
        // split. Sometimes there are features which cannot ensure this (e.g. a binary features are '1'). Thus, we iterate over a random permutation of features 
        // and return as soon as we find a valid split
        std::vector<unsigned int> features(X[0].size());
        std::iota(std::begin(features), std::end(features), 0); 

        std::mt19937 gen(seed);
        std::shuffle(features.begin(), features.end(), gen);

        for (auto const & f: features) {
            // We need to find the next smallest and next biggest value of the data to ensure that left/right will receive at-least 
            // one example. This is a brute force implementation in O(N)
            data_t smallest, second_smallest;
            if(X[0][f] <X[1][f]){
                smallest = X[0][f];
                second_smallest = X[1][f];
            } else {
                smallest = X[1][f];
                second_smallest = X[0][f];
            }

            data_t biggest, second_biggest;
            if(X[0][f] > X[1][f]){
                biggest = X[0][f];
                second_biggest = X[1][f];
            } else {
                biggest = X[1][f];
                second_biggest = X[0][f];
            }

            for (unsigned int i = 2; i < X.size(); ++i) {
                if(X[i][f] > smallest ) { 
                    second_smallest = smallest;
                    smallest = X[i][f];
                } else if(X[i][f] < second_smallest){
                    second_smallest = X[i][f];
                }

                if(X[i][f] > biggest ) { 
                    second_biggest = biggest;
                    biggest = X[i][f];
                } else if(X[i][f] > second_biggest){
                    second_biggest = X[i][f];
                }
            }

            // This is not a valid split if we cannot ensure that the left / right child receive at-least one example.
            if (second_smallest == smallest || second_biggest == biggest) continue;
            std::uniform_real_distribution<> fdis(second_smallest, second_biggest); 

            // So usually I would expect the following line to work, but for some reason it does not. Is this a gcc bug?
            //return std::make_pair<data_t, unsigned int>(static_cast<data_t>(fdis(gen)), f);
            int ftmp = f;
            return std::make_pair<data_t, unsigned int>(static_cast<data_t>(fdis(gen)), ftmp);
        }

        // If this is reached, then no valid split has been found and we default to the default split
        return std::make_pair<data_t, unsigned int>(1.0, 0);
    }

    /**
     * @brief  
     * @note   
     * @param  &X: 
     * @param  &Y: 
     * @retval None
     */
    void random_nodes(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        std::mt19937 gen(seed);
        std::uniform_int_distribution<> idis(0, X[0].size() - 1);
        std::uniform_real_distribution<> fdis(0,1);

        nodes.resize(n_nodes);
        for (unsigned int i = 0; i < n_nodes; ++i) {
            auto feature = idis(gen);
            //std::uniform_real_distribution<> fdis(amin(feature), amax(feature));
            nodes[i].feature = feature;
            nodes[i].threshold = fdis(gen);

            if (i >= start_leaf) {
                nodes[i].preds.resize(n_classes);
                std::fill(nodes[i].preds.begin(), nodes[i].preds.end(), 0);
                // for (unsigned int j = 0; n_classes; ++j) {
                //     nodes[i][j] = fdis(gen);
                // }
            } 
        }

        for (unsigned int i = 0; i < X.size(); ++i) {
            auto idx = node_index(X[i]);
            nodes[idx].preds[Y[i]] += 1;
            nodes[idx].total_cnt += 1;
        }

        for (unsigned int i = start_leaf; i < n_nodes; ++i) {
            auto & preds = nodes[i].preds;
            auto sum = nodes[i].total_cnt;
            if (nodes[i].total_cnt > 0) {
                std::transform(preds.begin(), preds.end(), preds.begin(), [sum](auto& c){return 1.0/sum*c;});
            } 
        }

        // for (unsigned int i = start_leaf; i < n_nodes; ++i) {
        //     auto & preds = nodes[i].preds;
        //     data_t sum = std::accumulate(preds.begin(), preds.end(),data_t(0));
        //     if (sum > 0) {
        //         std::transform(preds.begin(), preds.end(), preds.begin(), [sum](auto& c){return 1.0/sum*c;});
        //     } else {
        //         std::fill(preds.begin(), preds.end(), 1.0/n_classes);
        //     }
        // }
    }

    void trained_nodes(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, TREE_INIT tree_init) {
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

            std::pair<data_t, unsigned int> split;
            if (tree_init == TRAIN) {
                split = best_split(data.first, data.second, n_classes);
            } else {
                split = random_split(data.first, data.second, seed);
            }
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
            n.total_cnt = sum;
            for (auto const l : data.second) {
                n.preds[l] += 1;
            }

            if (sum > 0) {
                std::transform(n.preds.begin(), n.preds.end(), n.preds.begin(), [sum](auto& c){return 1.0/sum*c;});
            }

            // if (sum > 0) {
            //     std::transform(n.preds.begin(), n.preds.end(), n.preds.begin(), [sum](auto& c){return 1.0/sum*c;});
            // } else {
            //     std::fill(n.preds.begin(), n.preds.end(), 1.0/n_classes);
            // }
            nodes.push_back(n);
        }
    }

public:

    Tree(TREE_INIT tree_init, TREE_NEXT tree_next, unsigned int max_depth, unsigned int n_classes, unsigned long seed, std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) : n_classes(n_classes), seed(seed), tree_next(tree_next) {

        start_leaf = std::pow(2,max_depth) - 1;
        n_nodes = std::pow(2,max_depth + 1) - 1;

        if (tree_init == FULLY_RANDOM) {
            random_nodes(X, Y);
        } else {
            trained_nodes(X, Y, tree_init);
        }
    }

    /**
    Update trees:
        - incremental: update statistics in leaf nodes reached by current batch and improve it this way
        - gradient: perform gradient step
    **/
    void next(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, xt::xarray<data_t> const &tree_grad, data_t step_size, unsigned int tree_num) {
        if (tree_next == GRADIENT) {
            step_size = step_size / tree_grad.shape()[0];
            for (unsigned int i = 0; i < X.size(); ++i) {
                auto idx = node_index(X[i]);
                for (unsigned int j = 0; j < n_classes; ++j) {
                    nodes[idx].preds[j] = nodes[idx].preds[j] - step_size * tree_grad(tree_num, i, j);
                } 
            }
            /* Not implemented yet */
        } else if (tree_next == INCREMENTAL) {
            for (unsigned int i = 0; i < X.size(); ++i) {
                auto idx = node_index(X[i]);
                auto sum = nodes[idx].total_cnt;
                for (unsigned int j = 0; j < n_classes; ++j) {
                    if (j == Y[i]) {
                        nodes[idx].preds[j] = (sum * nodes[idx].preds[j] + 1) / (sum + 1);
                    } else {
                        nodes[idx].preds[j] = (sum * nodes[idx].preds[j]) / (sum + 1);
                    }
                }
                nodes[idx].total_cnt += 1;
            }
        } else {
            /* Do nothing */
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

    std::vector<std::vector<data_t>> predict_proba(std::vector<std::vector<data_t>> const &X) {
        std::vector<std::vector<data_t>> preds(X.size());
        for (unsigned int i = 0; i < X.size(); ++i) {
            std::vector<data_t> const & xpred = nodes[node_index(X[i])].preds;
            preds.push_back(xpred);
        }
        return preds;
    }
};

#endif