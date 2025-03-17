#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "AMGXSolver.h"

namespace py = pybind11;

class PyAMGXSolver {
public:
    AMGXSolver solver;

    PyAMGXSolver(const std::string& config_file, bool use_cpu = false,
                 std::vector<int> gpu_ids = {}, bool pin_memory = true,
                 const std::string& log_file = "") 
        : solver(config_file.c_str(), use_cpu, gpu_ids.empty() ? nullptr : gpu_ids.data(),
                 static_cast<int>(gpu_ids.size()), pin_memory, 
                 log_file.empty() ? nullptr : log_file.c_str()) {}

    void initializeMatrix(py::array_t<int> row_ptr, py::array_t<int> col_indices, 
                          py::array_t<double> values) {
        _num_rows = row_ptr.size() - 1;
        solver.initializeMatrix(num_rows, row_ptr.data(), col_indices.data(), values.data());
    }

    void replaceCoefficients(py::array_t<double> values) {
        solver.replaceCoefficients(_num_rows, values.size(), values.data());
    }

    py::array_t<double> solve(py::array_t<double> b) {
        py::array_t<double> x(b.size());
        solver.solve(x.mutable_data(), b.data(), b.size());
        return x;
    }
private:
    int _num_rows = 0;
};

// Bind to Python module
PYBIND11_MODULE(pyAMGXSolver, m) {
    py::class_<PyAMGXSolver>(m, "AMGXSolver")
        .def(py::init<const std::string&, bool, std::vector<int>, bool, const std::string&>(),
             py::arg("config_file"), py::arg("use_cpu") = false,
             py::arg("gpu_ids") = std::vector<int>{}, py::arg("pin_memory") = true,
             py::arg("log_file") = "")
        .def("initialize_matrix", &PyAMGXSolver::initializeMatrix)
        .def("replace_coefficients", &PyAMGXSolver::replaceCoefficients)
        .def("solve", &PyAMGXSolver::solve);
}
