#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include <pybind11/stl_bind.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"     // Numpy bindings

#include "BiasedProxEnsemble.h"

namespace py = pybind11;

PYBIND11_MODULE(PyBPE, m) {

py::class_<BiasedProxEnsemble<true>>(m, "RandomBiasedProxEnsemble")
    .def(py::init<unsigned int, unsigned int, unsigned long, data_t, data_t>(), py::arg("max_depth") = 10, py::arg("n_classes") = 1, py::arg("seed") = 1234, py::arg("alpha") = 0.01, py::arg("l_reg") = 0.001)
    .def ("next", &BiasedProxEnsemble<true>::next, py::arg("X"), py::arg("Y"))
    .def ("num_trees", &BiasedProxEnsemble<true>::num_trees)
    .def ("predict_proba", &BiasedProxEnsemble<true>::predict_proba, py::arg("X"));

py::class_<BiasedProxEnsemble<false>>(m, "TrainedBiasedProxEnsemble")
    .def(py::init<unsigned int, unsigned int, unsigned long, data_t, data_t>(), py::arg("max_depth") = 10, py::arg("n_classes") = 1, py::arg("seed") = 1234, py::arg("alpha") = 0.01, py::arg("l_reg") = 0.001)
    .def ("next", &BiasedProxEnsemble<false>::next, py::arg("X"), py::arg("Y"))
    .def ("num_trees", &BiasedProxEnsemble<false>::num_trees)
    .def ("predict_proba", &BiasedProxEnsemble<false>::predict_proba, py::arg("X"));
}