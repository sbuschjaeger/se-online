#ifndef REGULARIZER_H
#define REGULARIZER_H

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

enum REGULARIZER {NO,L0,L1};

inline data_t no_reg(xt::xarray<data_t> &w){
    return 0.0;
}

inline xt::xarray<data_t> no_prox(xt::xarray<data_t> &w, data_t step_size, data_t lambda){
    return w;
}

inline data_t L0_reg(xt::xarray<data_t> &w){
    data_t cnt = 0;
    for (unsigned int i = 0; i < w.shape()[0]; ++i) {
        cnt += (w(i) != 0.0);
    }

    return cnt;
}

inline xt::xarray<data_t> L0_prox(xt::xarray<data_t> &w, data_t step_size, data_t lambda){
    data_t tmp = std::sqrt(2 * lambda * step_size);
    for (unsigned int i = 0; i < w.shape()[0]; ++i) {
        if (std::abs(w(i)) < tmp)  {
            w(i) = 0.0;
        }
    }
    return w;
}

inline data_t L1_reg(xt::xarray<data_t> &w){
    data_t cnt = 0;
    for (unsigned int i = 0; i < w.shape()[0]; ++i) {
        cnt += std::abs(w(i));
    }

    return cnt;
}

inline xt::xarray<data_t> L1_prox(xt::xarray<data_t> &w, data_t step_size, data_t lambda){
    xt::xarray<data_t> sign = xt::sign(w);
    w = xt::abs(w) - step_size * lambda;
    return sign*xt::maximum(w,0);
}

auto reg_from_enum(REGULARIZER reg) {
    if (reg == REGULARIZER::NO) {
        return no_reg;
    } else if (reg == REGULARIZER::L0) {
        return L0_reg;
    } else if (reg == REGULARIZER::L1) {
        return L1_reg;
    } else {
        throw std::runtime_error("Wrong regularizer enum provided. No implementation for this enum found");
    }
}

auto prox_from_enum(REGULARIZER reg) {
    if (reg == REGULARIZER::NO) {
        return no_prox;
    } else if (reg == REGULARIZER::L0) {
        return L0_prox;
    } else if (reg == REGULARIZER::L1) {
        return L1_prox;
    } else {
        throw std::runtime_error("Wrong regularizer enum provided. No implementation for this enum found");
    }
}

auto regularizer_from_string(std::string const & regularizer) {
    if (regularizer == "none") {
        return REGULARIZER::NO;
    } else if (regularizer  == "L0") {
        return REGULARIZER::L0;
    } else if (regularizer == "L1") {
        return REGULARIZER::L1;
    } else {
        throw std::runtime_error("Currently only the three regularizer {none, L0, L1} are supported, but you provided: " + regularizer);
    }
}

#endif