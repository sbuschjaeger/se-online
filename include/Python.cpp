#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>

#include "BiasedProxEnsemble.h"

class BiasedProxEnsembleAdaptor {
private:
    BiasedProxedEnsembleInterface<data_t> * model = nullptr;

public:
    BiasedProxEnsembleAdaptor(
        unsigned int max_depth,
        unsigned int max_trees,
        unsigned int n_classes, 
        unsigned long seed, 
        data_t step_size,
        data_t lambda,
        data_t init_weight, 
        std::vector<bool> const &is_nominal,
        const std::string init_mode, 
        const std::string next_mode, 
        const std::string loss
    ) { 
        std::function< xt::xarray<data_t>(xt::xarray<data_t> const &, xt::xarray<data_t> const &) > _loss;
        std::function< xt::xarray<data_t>(xt::xarray<data_t> const &, xt::xarray<data_t> const &) > _loss_deriv;

        if (loss == "cross-entropy") {
            _loss = cross_entropy;
            _loss_deriv = cross_entropy_deriv;
        } else if (loss  == "mse") {
            _loss = mse;
            _loss_deriv = mse_deriv;
        } else if (loss == "hinge2") {
            _loss = hinge2;
            _loss_deriv = hinge2_deriv;
        } else {
            throw std::runtime_error("Currently only the three losses {cross-entropy, hinge2, mse} are supported, but you provided: " + loss);
        }

        // Yeha this is ugly and there is probably clever way to do this with C++17/20, but this was quicker to code and it gets the job done.
        // Also, lets be real here: There is only a limited chance more init/next modes are added without much refactoring of the whole project
        if (init_mode == "random" && next_mode == "incremental") {
            model = new BiasedProxEnsemble<TREE_INIT::RANDOM, TREE_NEXT::INCREMENTAL, data_t>(max_depth, max_trees, n_classes, seed, step_size, lambda, init_weight, is_nominal, _loss, _loss_deriv);
        } else if (init_mode == "random" && next_mode == "gradient") {
            model = new BiasedProxEnsemble<TREE_INIT::RANDOM, TREE_NEXT::GRADIENT, data_t>(max_depth, max_trees, n_classes, seed, step_size, lambda, init_weight,  is_nominal, _loss, _loss_deriv);
        } else if (init_mode == "random" && next_mode == "none") {
            model = new BiasedProxEnsemble<TREE_INIT::RANDOM, TREE_NEXT::NONE, data_t>(max_depth, max_trees, n_classes, seed, step_size, lambda, init_weight,  is_nominal, _loss, _loss_deriv);
        } else if (init_mode == "fully-random" && next_mode == "incremental") {
            model = new BiasedProxEnsemble<TREE_INIT::FULLY_RANDOM, TREE_NEXT::INCREMENTAL, data_t>(max_depth, max_trees, n_classes, seed, step_size, lambda, init_weight, is_nominal, _loss, _loss_deriv);
        } else if (init_mode == "fully-random" && next_mode == "gradient") {
            model = new BiasedProxEnsemble<TREE_INIT::FULLY_RANDOM, TREE_NEXT::GRADIENT, data_t>(max_depth, max_trees, n_classes, seed, step_size, lambda, init_weight, is_nominal, _loss, _loss_deriv);
        } else if (init_mode == "fully-random" && next_mode == "none") {
            model = new BiasedProxEnsemble<TREE_INIT::FULLY_RANDOM, TREE_NEXT::NONE, data_t>(max_depth, max_trees, n_classes, seed, step_size, lambda, init_weight, is_nominal,  _loss, _loss_deriv);
        } else if (init_mode == "train" && next_mode == "incremental") {
            model = new BiasedProxEnsemble<TREE_INIT::TRAIN, TREE_NEXT::INCREMENTAL, data_t>(max_depth, max_trees, n_classes, seed, step_size, lambda, init_weight, is_nominal,  _loss, _loss_deriv);
        } else if (init_mode == "train" && next_mode == "gradient") {
            model = new BiasedProxEnsemble<TREE_INIT::TRAIN, TREE_NEXT::GRADIENT, data_t>(max_depth, max_trees, n_classes, seed, step_size, lambda, init_weight, is_nominal,  _loss, _loss_deriv);
        } else if (init_mode == "train" && next_mode == "none") {
            model = new BiasedProxEnsemble<TREE_INIT::TRAIN, TREE_NEXT::NONE, data_t>(max_depth, max_trees, n_classes, seed, step_size, lambda, init_weight, is_nominal, _loss, _loss_deriv);
        } else {
            throw std::runtime_error("Currently only the three init_modes {random, fully-random, train} and the three next_modes {incremental, none, gradient} are supported for trees, but you provided a combination of " + init_mode + " and " + next_mode);
        }
    }

    data_t next(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        if (model != nullptr) {
            return model->next(X,Y);
        } else {
            return 0.0;
        }
    }

    std::vector<std::vector<data_t>> predict_proba(std::vector<std::vector<data_t>> const &X) {
        return model->predict_proba(X);
    }
    
    ~BiasedProxEnsembleAdaptor() {
        if (model != nullptr) {
            delete model;
        }
    }

    std::vector<data_t> weights() const {
        if (model != nullptr) {
            return model->weights();
        } else {
            return std::vector<data_t>();
        }
    }

    unsigned int num_trees() const {
        if (model != nullptr) {
            return model->num_trees();
        } else {
            return 0;
        }
    }
};

namespace py = pybind11;
PYBIND11_MODULE(PyBPE, m) {

py::class_<BiasedProxEnsembleAdaptor>(m, "BiasedProxEnsemble")
    .def(py::init<unsigned int, unsigned int, unsigned int, unsigned long, data_t, data_t, data_t, std::vector<bool>, std::string, std::string, std::string>(), py::arg("max_depth") = 10, py::arg("max_trees") = 0, py::arg("n_classes") = 1, py::arg("seed") = 1234, py::arg("step_size") = 0.01, py::arg("l_reg") = 0.001, py::arg("init_weight") = 0, py::arg("is_nominal"), py::arg("init_mode") = "random", py::arg("next_mode") = "incremental", py::arg("loss") = "cross-entropy")
    .def ("next", &BiasedProxEnsembleAdaptor::next, py::arg("X"), py::arg("Y"))
    .def ("num_trees", &BiasedProxEnsembleAdaptor::num_trees)
    .def ("predict_proba", &BiasedProxEnsembleAdaptor::predict_proba, py::arg("X"));
}