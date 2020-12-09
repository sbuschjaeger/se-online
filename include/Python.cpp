#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>

#include "BiasedProxEnsemble.h"

class BiasedProxEnsembleAdaptor {
private:
    BiasedProxEnsemble * model = nullptr;

public:
    BiasedProxEnsembleAdaptor(
        unsigned int max_depth,
        unsigned int max_trees,
        unsigned int n_classes, 
        unsigned long seed, 
        data_t step_size,
        data_t lambda,
        data_t init_weight, 
        const std::string mode, 
        const std::string loss
    ) { 
        TREE_TYPE tree_type;
        if (mode == "random") {
            tree_type = TREE_TYPE::RANDOM;
        } else if (mode == "fully-random") {
            tree_type = TREE_TYPE::FULLY_RANDOM;
        } else if (mode == "train") {
            tree_type = TREE_TYPE::TRAIN;
        } else {
            throw std::runtime_error("Currently only the three modes {random, fully-random, train} are supported for trees, but you provided: " + mode);
        }

        if (loss == "cross-entropy") {
            model = new BiasedProxEnsemble(max_depth, max_trees, n_classes, seed, step_size, lambda, init_weight, tree_type, cross_entropy, cross_entropy_deriv);
        } else {
            model = new BiasedProxEnsemble(max_depth, max_trees, n_classes, seed, step_size, lambda, init_weight, tree_type, mse, mse_deriv);
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
    .def(py::init<unsigned int, unsigned int, unsigned int, unsigned long, data_t, data_t, data_t, std::string, std::string>(), py::arg("max_depth") = 10, py::arg("max_trees") = 0, py::arg("n_classes") = 1, py::arg("seed") = 1234, py::arg("step_size") = 0.01, py::arg("l_reg") = 0.001, py::arg("init_weight") = 0, py::arg("mode") = "random", py::arg("loss") = "cross-entropy")
    .def ("next", &BiasedProxEnsembleAdaptor::next, py::arg("X"), py::arg("Y"))
    .def ("num_trees", &BiasedProxEnsembleAdaptor::num_trees)
    .def ("predict_proba", &BiasedProxEnsembleAdaptor::predict_proba, py::arg("X"));
}