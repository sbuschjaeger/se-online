#ifndef LOSSES_H
#define LOSSES_H

#include <vector>
#include <algorithm>
#include <math.h>
#include <numeric>
#include "Datatypes.h"

std::vector<data_t> softmax(std::vector<data_t> const &x) {
    std::vector<data_t> tmp(x);
    data_t m = *std::max_element(tmp.begin(), tmp.end());

    for(unsigned int i = 0; i < tmp.size(); i++) {
        tmp[i] = std::exp(tmp[i] - m);
    } 

    data_t sum = static_cast<data_t>(std::accumulate(tmp.begin(), tmp.end(), 0.0));
    std::transform(tmp.begin(), tmp.end(), tmp.begin(), [sum](data_t xi){ return xi/sum; });

    return tmp;
}

std::vector<data_t> cross_entropy(std::vector<data_t> const &pred, unsigned int target) {
    auto p = softmax(pred);
    for (unsigned int i = 0; i < p.size(); ++i) {
        if (i != target) {
            p[i] = 0;
        } else {
            p[i] = -std::log(p[i]);
        }
    }
    return p;
}

std::vector<data_t> cross_entropy_deriv(std::vector<data_t> const &pred, unsigned int target) {
    auto grad = softmax(pred);
    for (unsigned int i = 0; i < grad.size(); ++i) {
        if (i == target) {
            grad[i] -= 1;
        }
        // if (i != target) {
        //     grad[i] -= 0;
        // } else {
        //     grad[i] -= 1;
        // }
    }
    return grad;
}

// def cross_entropy(pred, target, epsilon=1e-12):
//     #pred = np.clip(pred, epsilon, 1.0 - epsilon)
//     p = softmax(pred, axis=1)
//     log_likelihood = -target*np.log(p)

//     return log_likelihood

// def cross_entropy_deriv(pred, target):
//     m = target.shape[0]
//     grad = softmax(pred, axis=1)
//     grad[range(m),target.argmax(axis=1)] -= 1
//     #grad = grad/m
//     return grad

#endif