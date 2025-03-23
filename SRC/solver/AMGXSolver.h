/**
 * @file AMGXSolver.h
 * @brief Header file for NVIDIA AMGX solver wrapper class
 *
 * This file defines a C++ wrapper class for the NVIDIA AMGX library,
 * providing a simplified interface for solving large sparse linear systems
 * using GPU acceleration. The class manages AMGX resources and provides
 * exception-safe operations.
 */

#ifndef AMGX_h
#define AMGX_h

#include <amgx_c.h>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cstring>

// Define SOLVER_API macro for cross-platform shared library export
#ifdef _WIN32
    #ifdef BUILDING_SOLVER
        #define SOLVER_API __declspec(dllexport)
    #else
        #define SOLVER_API __declspec(dllimport)
    #endif
#else
    #define SOLVER_API
#endif

/**
 * @brief Macro for AMGX error checking
 *
 * Wraps AMGX function calls with error checking. Throws std::runtime_error
 * with detailed error message if the AMGX call fails.
 *
 * @param rc The AMGX function call to check
 * @throws std::runtime_error if the AMGX call returns an error
 */
#define CHECK_AMGX_CALL(rc) \
{ \
  AMGX_RC err;     \
  char msg[4096];   \
  switch(err = (rc)) {    \
  case AMGX_RC_OK: \
    break; \
  default: \
    AMGX_get_error_string(err, msg, 4096); \
    fprintf(stderr, "AMGX ERROR in function: %s at file %s line %6d\n", #rc, __FILE__, __LINE__); \
    fprintf(stderr, "AMGX ERROR: %s\n", msg); \
    throw std::runtime_error(std::string("AMGX ERROR: ") + msg); \
  } \
}

/**
 * @brief C++ wrapper class for NVIDIA AMGX solver
 *
 * Provides a high-level interface to the AMGX library for solving sparse
 * linear systems. Manages AMGX resources and handles initialization,
 * matrix setup, solving, and cleanup operations.
 */
class SOLVER_API AMGXSolver
{
public:
    /**
     * @brief Construct a new AMGXSolver
     *
     * @param config_file Path to AMGX configuration JSON file
     * @param use_cpu Whether to use CPU instead of GPU
     * @param gpu_ids Array of GPU device IDs to use
     * @param num_gpus Number of GPUs to use
     * @param pin_memory Whether to use pinned memory
     * @param log_file Path to log file (optional)
     */
    AMGXSolver(const char* config_file, bool use_cpu = false, const int* gpu_ids = nullptr,
               int num_gpus = 0, bool pin_memory = false, const char* log_file = nullptr);
    
    /**
     * @brief Destroy the AMGXSolver and clean up resources
     */
    ~AMGXSolver();
    
    /**
     * @brief Initialize the solver with given configuration
     *
     * @param config_file Path to AMGX configuration JSON file
     * @param use_cpu Whether to use CPU instead of GPU
     * @param gpu_ids Array of GPU device IDs to use
     * @param num_gpus Number of GPUs to use
     * @param pin_memory Whether to use pinned memory
     * @param log_file Path to log file (optional)
     */
    void initialize(const char *config_file, bool use_cpu, const int *gpu_ids, 
                   int num_gpus, bool pin_memory, const char* log_file);

    /**
     * @brief Clean up all AMGX resources
     */
    void cleanup();

    /**
     * @brief Initialize the sparse matrix in CSR format
     *
     * @param num_rows Number of rows in the matrix
     * @param row_ptr CSR row pointers array
     * @param col_indices CSR column indices array
     * @param values Matrix values array
     */
    void initializeMatrix(int num_rows, const int* row_ptr, const int* col_indices,
                         const double* values);

    /**
     * @brief Replace matrix coefficients while keeping the same sparsity pattern
     *
     * @param num_rows Number of rows in the matrix
     * @param num_non_zeros Number of non-zero elements
     * @param values New matrix values array
     */
    void replaceCoefficients(int num_rows, int num_non_zeros, 
                           const double *values);

    /**
     * @brief Solve the system using previously set vectors
     *
     * @return Solver status (0 = success)
     */
    int solve(void);

    /**
     * @brief Solve the system Ax = b
     *
     * @param x Solution vector (output)
     * @param b Right-hand side vector
     * @param num_rows Number of rows in the system
     * @return Solver status (0 = success)
     */
    int solve(double* x, const double* b, int num_rows);

    /**
     * @brief Get the number of iterations performed in last solve
     *
     * @return Number of iterations
     */
    int getNumIterations(void);

    /**
     * @brief Get the final residual norm from last solve
     *
     * @return Final residual norm
     */
    double getFinalResidual(void);

private:
    /**
     * @brief Callback function for AMGX logging
     *
     * @param msg Message to log
     * @param length Length of the message
     */
    static void callback(const char* msg, int length);

    // Static members for global state
    static std::ofstream _log_file_stream;   ///< Log file stream
    static bool _use_log_file;               ///< Whether logging is enabled
    static bool _amgx_initialized;           ///< Whether AMGX is initialized
    static int _active_solver_instances;     ///< Count of active solver instances

    // AMGX handles
    AMGX_config_handle    _config       = nullptr;  ///< Configuration handle
    AMGX_resources_handle _resources    = nullptr;  ///< Resources handle
    AMGX_matrix_handle    _matrix       = nullptr;  ///< Matrix handle
    AMGX_vector_handle    _rhs          = nullptr;  ///< Right-hand side vector handle
    AMGX_vector_handle    _solution     = nullptr;  ///< Solution vector handle
    AMGX_solver_handle    _solver       = nullptr;  ///< Solver handle
    AMGX_Mode             _mode;                    ///< Solver mode

    // Configuration flags
    bool _use_cpu;      ///< Whether to use CPU instead of GPU
    bool _pin_memory;   ///< Whether to use pinned memory
    
    // Matrix dimensions
    int _num_rows = 0;        ///< Number of rows in the matrix
    int _num_non_zeros = 0;   ///< Number of non-zero elements
};

#endif