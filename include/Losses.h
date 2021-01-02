#ifndef LOSSES_H
#define LOSSES_H

#include <vector>
#include <algorithm>
#include <math.h>
#include <numeric>

#include "xtensor/xstrided_view.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xadapt.hpp"

#include "Datatypes.h"

/**
 * @brief  The softmax function which maps the input tensor to probabilities. The shape is assumed to be (batch_size, n_classes). Softmax is applied for each row of the input matrix.
 * @note   
 * @param  &X: Inputmatrix over which softmax will be applied. Assumed to have shape (batch_size, n_classes) 
 * @retval A new matrix with shape (batch_size, n_classes) where softmax has been applied to every row.
 */
xt::xarray<data_t> softmax(xt::xarray<data_t> const &X) {
    xt::xarray<data_t> tmp = xt::xarray<data_t>(X);
    int batch_size = tmp.shape()[0];
    tmp = xt::exp(tmp - xt::reshape_view(xt::amax(tmp, 1), { batch_size, 1 }));
    return tmp / xt::reshape_view(xt::sum(tmp, 1), { batch_size, 1 }); 
}

/**
 * @brief  Computes the cross entropy loss, which is the negative log-liklihood combined with the softmax function. The prediction tensor is assumed to have a shape of (batch_size, n_classes), whereas the target vector is assumed to be a vector of shape (batch_size) in which each entry represents the corresponding class.
 * @note   
 * @param  &pred: The per-class prediction tensor for each sample. This tensor is assumed to have a shape of (batch_size, n_classes)
 * @param  &target: The per-sample target which is assumed to be from {0,\dots,n_classes - 1}. This tensor is assumed to have a shape of (batch_size)
 * @retval The cross-entropy for each class and each sample. The return tensor has a shape of (batch_size, n_classes). 
 */
xt::xarray<data_t> cross_entropy(xt::xarray<data_t> const &pred, xt::xarray<unsigned int> const &target){
    xt::xarray<data_t> p = softmax(pred); 

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

/**
 * @brief  The first derivation of the cross entropy loss. The prediction tensor is assumed to have a shape of (batch_size, n_classes), whereas the target vector is assumed to be a vector of shape (batch_size) in which each entry represents the corresponding class.
 * @note   
 * @param  &pred: The per-class prediction tensor for each sample. This tensor is assumed to have a shape of (batch_size, n_classes)
 * @param  &target: The per-sample target which is assumed to be from {0,\dots,n_classes - 1}. This tensor is assumed to have a shape of (batch_size)
 * @retval The first derivation of the cross-entropy for each class and each sample. The return tensor has a shape of (batch_size, n_classes). 
 */
xt::xarray<data_t> cross_entropy_deriv(xt::xarray<data_t> const &pred, xt::xarray<unsigned int> const &target){
    xt::xarray<data_t> grad = softmax(pred);

    for (unsigned int i = 0; i < pred.shape()[0]; ++i) {
        grad(i, target(i)) -= 1.0; 
    }

    return grad;
}

/**
 * @brief  Computes the mse loss. Contrary to common implementations of the mse this version first scales the input to probabilities using the softmax!
 * @note   This is not the regular MSE loss, but it maps the prediction to probabilities beforehand using softmax!
 * @param  &pred: The per-class prediction tensor for each sample. This tensor is assumed to have a shape of (batch_size, n_classes)
 * @param  &target: The per-sample target which is assumed to be from {0,\dots,n_classes - 1}. This tensor is assumed to have a shape of (batch_size)
 * @retval The mse loss for each class and each sample. The return tensor has a shape of (batch_size, n_classes). 
 */
xt::xarray<data_t> mse(xt::xarray<data_t> const &pred, xt::xarray<unsigned int> const &target){
    //xt::xarray<data_t> p = softmax(pred);
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

    //return (p-target_one_hot)*(p-target_one_hot);
    return (pred-target_one_hot)*(pred-target_one_hot);
}

/**
 * @brief  Computes the first derivative of the mse loss. Contrary to common implementations of the mse, this version first scales the input to probabilities using the softmax!
 * @note   This is not the regular MSE loss, but it maps the prediction to probabilities beforehand using softmax!
 * @param  &pred: The per-class prediction tensor for each sample. This tensor is assumed to have a shape of (batch_size, n_classes)
 * @param  &target: The per-sample target which is assumed to be from {0,\dots,n_classes - 1}. This tensor is assumed to have a shape of (batch_size)
 * @retval The first derivative of the mse loss for each class and each sample. The return tensor has a shape of (batch_size, n_classes). 
 */
xt::xarray<data_t> mse_deriv(xt::xarray<data_t> const &pred, xt::xarray<data_t> const &target){
    //xt::xarray<data_t> p = softmax(pred);
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

    //return 2 * (pred - target_one_hot) * p * (1.0 - p) ;
    return 2 * (pred - target_one_hot) ;
}

xt::xarray<data_t> hinge2(xt::xarray<data_t> const &pred, xt::xarray<data_t> const &target){
    xt::xarray<data_t> losses = xt::xarray<data_t>::from_shape(pred.shape());

    for (unsigned int i = 0; i < pred.shape()[0]; ++i) {
        for (unsigned int j = 0; j < pred.shape()[1]; ++j) {
            if (target(i) == j) {
                losses(i,j) = std::max(1.0 - 1 * pred(i,j), 0.0) * std::max(1.0 - 1 * pred(i,j), 0.0);
            } else {
                losses(i,j) = std::max(1.0 - (-1 * pred(i,j)), 0.0) * std::max(1.0 - (-1 * pred(i,j)), 0.0);
            }
        }
    }

    return losses;
}

xt::xarray<data_t> hinge2_deriv(xt::xarray<data_t> const &pred, xt::xarray<data_t> const &target){
    xt::xarray<data_t> losses_deriv = xt::xarray<data_t>::from_shape(pred.shape());

    for (unsigned int i = 0; i < pred.shape()[0]; ++i) {
        for (unsigned int j = 0; j < pred.shape()[1]; ++j) {
            if (target(i) == j) {
                losses_deriv(i,j) = 2 * std::max(1.0 - 1 * pred(i,j), 0.0);
            } else {
                losses_deriv(i,j) = 2 * std::max(1.0 - (-1 * pred(i,j)), 0.0);
            }
        }
    }

    return losses_deriv;
}

#endif