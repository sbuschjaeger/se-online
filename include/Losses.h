#ifndef LOSSES_H
#define LOSSES_H

#include <vector>
#include <algorithm>
#include <math.h>
#include <numeric>

#include "xtensor/xstrided_view.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xadapt.hpp"

#include "Datatypes.h"

xt::xarray<data_t> softmax(xt::xarray<data_t> const &X) {
    xt::xarray<data_t> tmp = xt::xarray<data_t>(X);
    int batch_size = tmp.shape()[0];
    tmp = xt::exp(tmp - xt::reshape_view(xt::amax(tmp, 1), { batch_size, 1 }));
    return tmp / xt::reshape_view(xt::sum(tmp, 1), { batch_size, 1 }); //xt::sum(tmp, 1);
}

xt::xarray<data_t> cross_entropy(xt::xarray<data_t> const &pred, xt::xarray<data_t> const &target){
    //TODO Assert shape
    xt::xarray<data_t> p = softmax(pred); //#, axis=1)

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
    
    return -target_one_hot*xt::log(p);
}

xt::xarray<data_t> cross_entropy_deriv(xt::xarray<data_t> const &pred, xt::xarray<data_t> const &target){
    //TODO Assert shape
    xt::xarray<data_t> grad = softmax(pred);

    for (unsigned int i = 0; i < pred.shape()[0]; ++i) {
        grad(i, target(i)) -= 1.0; 
    }

    return grad;
}

xt::xarray<data_t> mse(xt::xarray<data_t> const &pred, xt::xarray<data_t> const &target){
    //TODO Assert shape
    xt::xarray<data_t> target_one_hot = xt::xarray<data_t>::from_shape(pred.shape());
    xt::xarray<data_t> scaled_pred = xt::xarray<data_t>::from_shape(pred.shape());
    xt::xarray<data_t> sums = xt::sum(pred, 1);

    // std:: cout << "sums: ";
    // for (unsigned int i = 0; i < sums.shape()[0]; ++i) {
    //     std:: cout << sums(i) << " ";
    // }
    // std::cout << std::endl;
    // std::cout << sums << std::endl;

    for (unsigned int i = 0; i < pred.shape()[0]; ++i) {
        for (unsigned int j = 0; j < pred.shape()[1]; ++j) {
            if (target(i) == j) {
                target_one_hot(i,j) = 1;
            } else {
                target_one_hot(i,j) = 0;
            }
            if (sums(i) > 0){
                scaled_pred(i,j) = pred(i,j) / sums(i); 
            } else {
                scaled_pred(i,j) = pred(i,j);
            }
        }
    }

    // std:: cout << "scaled_pred: ";
    // for (unsigned int i = 0; i < sums.shape()[0]; ++i) {
    //     std:: cout << scaled_pred(i, 0) << " ";
    // }
    // std::cout << std::endl;

    return (scaled_pred-target_one_hot)*(scaled_pred-target_one_hot);
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

#endif