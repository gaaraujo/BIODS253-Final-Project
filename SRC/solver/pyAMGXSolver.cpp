/**
 * @file pyAMGXSolver.cpp
 * @brief Python bindings for the AMGX solver using pybind11
 *
 * This module provides Python bindings for the AMGX solver library, allowing Python
 * users to leverage NVIDIA's AMGX for solving large sparse linear systems on GPUs.
 * The implementation wraps the C++ AMGXSolver class and handles conversion between
 * Python/NumPy arrays and C++ data structures.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "AMGXSolver.h"

namespace py = pybind11;

/**
 * @brief Python wrapper class for AMGXSolver
 *
 * Provides a Python-friendly interface to the C++ AMGXSolver class, handling
 * memory management and NumPy array conversions. Implements context manager
 * protocol for safe resource handling in Python.
 */
class PyAMGXSolver {
public:
    AMGXSolver* solver;  // Pointer to the underlying C++ solver
    int _num_rows;       // Number of rows in the matrix

    /**
     * @brief Construct a new PyAMGXSolver
     *
     * @param config_file Path to AMGX configuration JSON file
     * @param use_cpu Whether to use CPU instead of GPU
     * @param gpu_ids List of GPU device IDs to use
     * @param pin_memory Whether to use pinned memory
     * @param log_file Path to log file (optional)
     * @throws std::runtime_error if initialization fails
     */
    PyAMGXSolver(const std::string& config_file, bool use_cpu = false,
                 std::vector<int> gpu_ids = {}, bool pin_memory = false,
                 py::object log_file = py::none()) 
        : solver(nullptr), _num_rows(0) 
    {
        try {
            solver = new AMGXSolver(
                config_file.c_str(),
                use_cpu,
                gpu_ids.empty() ? nullptr : gpu_ids.data(),
                static_cast<int>(gpu_ids.size()),
                pin_memory,
                log_file.is_none() ? nullptr : log_file.cast<std::string>().c_str()
            );
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Failed to initialize PyAMGXSolver: " << e.what() << std::endl;
            cleanup();  // Ensure proper cleanup
            throw;  // Rethrow exception
        }
    }

    /**
     * @brief Destructor ensures proper cleanup of resources
     */
    ~PyAMGXSolver() {
        cleanup();
    }

    /**
     * @brief Initialize the sparse matrix in CSR format
     *
     * @param row_ptr_obj Row pointers array (numpy.ndarray[int32])
     * @param col_indices_obj Column indices array (numpy.ndarray[int32])
     * @param values_obj Matrix values array (numpy.ndarray[float64])
     * @throws std::invalid_argument if inputs are invalid
     * @throws std::runtime_error if solver is not initialized
     */
    void initializeMatrix(py::object row_ptr_obj, py::object col_indices_obj, py::object values_obj) 
    {
        if (!solver) {
            throw std::runtime_error("Solver is not initialized.");
        }

        // Explicitly check for Python `None` before converting to `py::array_t<>`
        if (row_ptr_obj.is_none() || col_indices_obj.is_none() || values_obj.is_none()) {
            throw std::invalid_argument("initializeMatrix: row_ptr, col_indices, and values cannot be None.");
        }

        // Convert to numpy arrays
        auto row_ptr = row_ptr_obj.cast<py::array_t<int>>();
        auto col_indices = col_indices_obj.cast<py::array_t<int>>();
        auto values = values_obj.cast<py::array_t<double>>();

        // Ensure they are non-empty
        if (row_ptr.size() == 0 || col_indices.size() == 0 || values.size() == 0) {
            throw std::invalid_argument("initializeMatrix: row_ptr, col_indices, and values must not be empty.");
        }

        // Ensure valid pointers
        if (!row_ptr.data() || !col_indices.data() || !values.data()) {
            throw std::invalid_argument("initializeMatrix: row_ptr, col_indices, and values must be valid numpy arrays.");
        }

        _num_rows = row_ptr.size() - 1;

        try {
            solver->initializeMatrix(_num_rows, row_ptr.data(), col_indices.data(), values.data());
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Exception in initializeMatrix: " << e.what() << std::endl;
            cleanup();  // Clean up resources to prevent leaks
            throw;
        }
    }

    /**
     * @brief Replace matrix coefficients while keeping the same sparsity pattern
     *
     * @param values_obj New matrix values array (numpy.ndarray[float64])
     * @throws std::invalid_argument if values array is invalid
     * @throws std::runtime_error if solver is not initialized
     */
    void replaceCoefficients(py::object values_obj) {
        if (!solver) {
            throw std::runtime_error("Solver is not initialized.");
        }

        // Explicitly check for Python `None`
        if (values_obj.is_none()) {
            throw std::invalid_argument("replaceCoefficients: values cannot be None.");
        }

        // Convert Python object to NumPy array
        auto values = values_obj.cast<py::array_t<double>>();

        // Ensure the array is non-empty
        if (values.size() == 0) {
            throw std::invalid_argument("replaceCoefficients: values cannot be empty.");
        }

        // Ensure a valid pointer
        if (!values.data()) {
            throw std::invalid_argument("replaceCoefficients: values must be a valid NumPy array.");
        }

        try {
            solver->replaceCoefficients(_num_rows, values.size(), values.data());
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Exception in replaceCoefficients: " << e.what() << std::endl;
            cleanup();  // Prevent memory leaks if an exception occurs
            throw;
        }
    }

    /**
     * @brief Solve the linear system Ax = b
     *
     * @param b_obj Right-hand side vector (numpy.ndarray[float64])
     * @return tuple containing:
     *         - Solution vector x (numpy.ndarray[float64])
     *         - Solver status (int, 0 = success)
     *         - Number of iterations performed (int)
     *         - Final residual norm (double)
     * @throws std::invalid_argument if input is invalid
     * @throws std::runtime_error if solver is not initialized
     */
    std::tuple<py::array_t<double>, int, int, double> solve(py::object b_obj) {
        if (!solver) {
            throw std::runtime_error("Solver is not initialized.");
        }

        // Explicitly check if `b_obj` is `None`
        if (b_obj.is_none()) {
            throw std::invalid_argument("solve: vector b cannot be None.");
        }

        // Convert Python object to NumPy array
        auto b = b_obj.cast<py::array_t<double>>();

        // Ensure the array is non-empty
        if (b.size() == 0) {
            throw std::invalid_argument("solve: vector b cannot be empty.");
        }

        // Ensure a valid pointer
        if (!b.data()) {
            throw std::invalid_argument("solve: vector b must be a valid NumPy array.");
        }

        // Allocate output array
        py::array_t<double> x(b.size());

        int solve_status = -3;
        int num_iterations = 0;
        double final_residual = 0.0;

        try {
            solve_status = solver->solve(x.mutable_data(), b.data(), b.size());
            num_iterations = solver->getNumIterations();
            final_residual = solver->getFinalResidual();
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Exception in solve: " << e.what() << std::endl;
            cleanup();  // Ensure proper cleanup if error occurs
            throw;
        }

        // Return solution, solve status, number of iterations, and final residual
        return std::make_tuple(x, solve_status, num_iterations, final_residual);
    }

    /**
     * @brief Clean up solver resources
     *
     * Frees memory and resets the solver pointer. Safe to call multiple times.
     */
    void cleanup() {
        if (solver) {
            delete solver;
            solver = nullptr;
        }
    }
};

/**
 * @brief Define the Python module and its interface
 *
 * Creates the Python module 'pyAMGXSolver' and exposes the AMGXSolver class
 * with its methods. Implements context manager protocol for use with
 * Python's 'with' statement.
 */
PYBIND11_MODULE(pyAMGXSolver, m) {
    // Create Python class binding
    py::class_<PyAMGXSolver>(m, "AMGXSolver")
        // Constructor with default arguments
        .def(py::init<const std::string&, bool, std::vector<int>, bool, py::object>(),
             py::arg("config_file"),
             py::arg("use_cpu") = false,
             py::arg("gpu_ids") = std::vector<int>{},
             py::arg("pin_memory") = false,
             py::arg("log_file") = py::none())
        // Expose methods
        .def("initialize_matrix", &PyAMGXSolver::initializeMatrix)
        .def("replace_coefficients", &PyAMGXSolver::replaceCoefficients)
        .def("solve", &PyAMGXSolver::solve)
        .def("cleanup", &PyAMGXSolver::cleanup)
        // Context manager support
        .def("__enter__", [](PyAMGXSolver &self) { return &self; })
        .def("__exit__", [](PyAMGXSolver &self, py::handle, py::handle, py::handle) {
            self.cleanup();
        });
}
