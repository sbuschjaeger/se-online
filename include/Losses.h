#ifndef LOSSES_H
#define LOSSES_H

#include <vector>
#include <algorithm>
#include <math.h>
#include <numeric>

#include "xtensor/xstrided_view.hpp"
#include "xtensor/xsort.hpp"

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

xt::xarray<data_t> softmax(xt::xarray<data_t> const &X) {
    xt::xarray<data_t> tmp = xt::xarray<data_t>(X);
    int batch_size = tmp.shape()[0];
    tmp = xt::exp(tmp - xt::reshape_view(xt::amax(tmp, 1), { batch_size, 1 }));
    return tmp / xt::reshape_view(xt::sum(tmp, 1), { batch_size, 1 }); //xt::sum(tmp, 1);
    // auto m = xt::argmax(tmp, 0);
    // tmp = xt::exp(tmp - m);
    // tmp = tmp / xt::sum(tmp, 1);
    // return tmp;
}

xt::xarray<data_t> cross_entropy(xt::xarray<data_t> const &pred, xt::xarray<data_t> const &target){
    xt::xarray<data_t> p = softmax(pred); //#, axis=1)
    xt::xarray<data_t> tmp = -target*xt::log(p);
    return -target*xt::log(p);
}

xt::xarray<data_t> cross_entropy_deriv(xt::xarray<data_t> const &pred, xt::xarray<data_t> const &target){
    //auto m = target.shape()[0];
    xt::xarray<data_t> grad = softmax(pred); //, axis=1
    auto amax = xt::argmax(target, 1);

    for (unsigned int i = 0; i < amax.shape()[0]; ++i) {
        grad(i, amax(i)) -= 1.0; 
    }

    // grad[range(m),xt::argmax(target, 1)] -= 1
    return grad;
}

xt::xarray<data_t> mse(xt::xarray<data_t> const &pred, xt::xarray<data_t> const &target){
    //TODO Assert shape
    xt::xarray<data_t> target_one_hot = xt::xarray<data_t>::from_shape(pred.shape());

    for (unsigned int i = 0; i < pred.shape()[0]; ++i) {
        for (unsigned int j = 0; j < pred.shape()[1]; ++j) {
            if (target(i) == j) {
                target_one_hot(i,j) = 1;
            } else {
                target_one_hot(i,j) = 0;
            }
        }
    }

    return (pred-target_one_hot)*(pred-target_one_hot);
}

xt::xarray<data_t> mse_deriv(xt::xarray<data_t> const &pred, xt::xarray<data_t> const &target){
    //TODO Assert shape
    xt::xarray<data_t> target_one_hot = xt::xarray<data_t>::from_shape(pred.shape());

    for (unsigned int i = 0; i < pred.shape()[0]; ++i) {
        for (unsigned int j = 0; j < pred.shape()[1]; ++j) {
            if (target(i) == j) {
                target_one_hot(i,j) = 1;
            } else {
                target_one_hot(i,j) = 0;
            }
        }
    }

    return 2 * (pred - target_one_hot) ;
}

// auto cross_entropy(xt::xarray<data_t> const &pred, xt::xarray<data_t> const &targets) {
//     auto p = softmax(pred);
//     for (unsigned int i = 0; i < p.size(); ++i) {
//         if (i != target) {
//             p[i] = 0;
//         } else {
//             p[i] = -std::log(p[i]);
//         }
//     }
//     return p;
// }

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