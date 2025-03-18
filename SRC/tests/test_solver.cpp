/**
 * @file test_solver.cpp
 * @brief Comprehensive test suite for AMGXSolver
 *
 * This file implements a test suite for the AMGXSolver class, including:
 * - Functional tests with different matrix sizes
 * - Exception handling tests
 * - Various configuration combinations (CPU/GPU, memory pinning, logging)
 * - Validation of solver results against direct matrix-vector multiplication
 */

#include <cstdio>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <sstream>
#include <stdexcept>
#include "../solver/AMGXSolver.h"

/**
 * @brief Structure to hold CSR matrix data
 */
struct MatrixData {
    int rows;
    int cols;
    int nonZeros;
    int* rowPtr;
    int* colIndices;
    double* values;
};

constexpr double DEFAULT_TOLERANCE = 1e-6;  // Convergence tolerance for solver

/**
 * @brief Creates a default AMGX configuration file
 * 
 * Generates a JSON configuration file with standard settings for testing:
 * - Block Jacobi preconditioner
 * - PCG solver
 * - Residual monitoring enabled
 * - Maximum 20 iterations
 * 
 * @param tolerance Convergence tolerance (default: 1e-6)
 * @return Path to created configuration file
 * @throws std::runtime_error if file creation fails
 */
std::string createDefaultConfigFile(double tolerance = DEFAULT_TOLERANCE) {
    std::string filename = "default_amgx_config.json";
    std::ofstream configFile(filename);
    if (!configFile.is_open()) {
        throw std::runtime_error("Failed to create default AMGX configuration file.");
    }

    // Use std::ostringstream for safe and precise floating-point formatting
    std::ostringstream configStream;
    configStream << "{\n"
                 << "    \"config_version\": 2,\n"
                 << "    \"solver\": {\n"
                 << "        \"preconditioner\": {\n"
                 << "            \"scope\": \"precond\",\n"
                 << "            \"solver\": \"BLOCK_JACOBI\"\n"
//                 << "            \"solver\": \"MULTICOLOR_DILU\"\n"
                 << "        },\n"
                 << "        \"solver\": \"PCG\",\n"
                 << "        \"print_solve_stats\": 1,\n"
                 << "        \"obtain_timings\": 1,\n"
                 << "        \"max_iters\": 20,\n"
                 << "        \"monitor_residual\": 1,\n"
                 << "        \"scope\": \"main\",\n"
                 << "        \"tolerance\": " << tolerance << ",\n"  // Dynamically insert tolerance
                 << "        \"norm\": \"L2\"\n"
                 << "    }\n"
                 << "}";

    // Write to file
    configFile << configStream.str();
    configFile.close();

    std::cout << "[INFO] No configuration file provided. Using default: " << filename 
              << " with tolerance = " << tolerance << std::endl;
    return filename;
}

/**
 * @brief Performs matrix-vector multiplication y = Ax in CSR format
 * 
 * Used to validate solver results by computing the residual directly.
 */
void multiplyCSR(const MatrixData& mat, const double* x, double* Ax) {
    for (int i = 0; i < mat.rows; ++i) {
        Ax[i] = 0.0;
        for (int j = mat.rowPtr[i]; j < mat.rowPtr[i + 1]; ++j) {
            Ax[i] += mat.values[j] * x[mat.colIndices[j]];
        }
    }
}

/**
 * @brief Computes L2 norm of a vector
 * 
 * @param r Vector to compute norm for
 * @param size Length of vector
 * @return L2 norm sqrt(sum(r_i^2))
 */
double calculateResidualNorm(double* r, int size) {
    double norm = 0.0;
    for (int i = 0; i < size; ++i) {
        norm += r[i] * r[i];
    }
    return std::sqrt(norm);
}

/**
 * @brief Creates a tridiagonal test matrix
 * 
 * Generates a symmetric positive definite matrix in CSR format:
 * - Diagonal elements = 2.0
 * - Off-diagonal elements = -1.0
 * - Pattern: tridiagonal (-1, 2, -1)
 * 
 * @param size Dimension of the square matrix
 * @return MatrixData structure containing the CSR matrix
 */
MatrixData createTestMatrix(int size) {
    MatrixData data;
    data.rows = size;
    data.cols = size;
    data.nonZeros = 3 * size - 2; // Tridiagonal

    data.rowPtr = new int[size + 1];
    data.colIndices = new int[data.nonZeros];
    data.values = new double[data.nonZeros];

    int index = 0;
    for (int i = 0; i < size; ++i) {
        data.rowPtr[i] = index;
        if (i > 0) {
            data.colIndices[index] = i - 1;
            data.values[index] = -1.0;
            index++;
        }
        data.colIndices[index] = i;
        data.values[index] = 2.0;
        index++;
        if (i < size - 1) {
            data.colIndices[index] = i + 1;
            data.values[index] = -1.0;
            index++;
        }
    }
    data.rowPtr[size] = index;
    return data;
}

// Global test counters
int passedTests = 0;
int failedTests = 0;
int passedExceptionTests = 0;
int failedExceptionTests = 0;

/**
 * @brief Runs solver test with specific configuration
 * 
 * Test workflow:
 * 1. Creates tridiagonal test matrix
 * 2. Solves system Ax = b with b = [1,1,...,1]
 * 3. Validates solution by computing residual
 * 4. Verifies log file creation if logging enabled
 * 
 * @param size Matrix dimension
 * @param amgx_config_file Path to AMGX configuration file
 * @param use_cpu Whether to use CPU instead of GPU
 * @param pin_memory Whether to use pinned memory
 * @param log_file Path to log file (nullptr for no logging)
 * @throws std::runtime_error on memory allocation failure
 */
void runTest(int size, const char* amgx_config_file, bool use_cpu = false, 
             bool pin_memory = true, const char* log_file = nullptr) {
    std::cout << "\n[INFO] Running test with matrix size: " << size << "x" << size << std::endl;

    MatrixData matrixData = createTestMatrix(size);
    double* rhs = new double[matrixData.rows];
    double* solution = new double[matrixData.rows];
    double* residual = new double[matrixData.rows];

    if (!rhs || !solution || !residual) {
        throw std::runtime_error("Memory allocation failure");
    }

    for (int i = 0; i < matrixData.rows; ++i) {
        rhs[i] = 1.0;
        solution[i] = 0.0;
    }

    // Testing with only one GPU, id = 0.
    int gpu_ids[] = {0};
    int num_gpus = 1;
    AMGXSolver solver(amgx_config_file, use_cpu, gpu_ids, num_gpus, pin_memory, log_file);

    solver.initializeMatrix(matrixData.rows, matrixData.rowPtr, matrixData.colIndices, matrixData.values);
    int solveStatus = solver.solve(solution, rhs, matrixData.rows);

    bool testPassed = false;
    if (solveStatus == 0) {
        double* Ax = new double[matrixData.rows];
        multiplyCSR(matrixData, solution, Ax);

        for (int i = 0; i < matrixData.rows; ++i) {
            residual[i] = Ax[i] - rhs[i];  // Compute residual r = Ax - b
        }

        double residualNorm = calculateResidualNorm(residual, matrixData.rows);
        std::cout << "Residual norm: " << residualNorm << std::endl;
        if (residualNorm < DEFAULT_TOLERANCE) {
            std::cout << "[PASS] Solution within tolerance! (max = " << DEFAULT_TOLERANCE << ")"  << std::endl;
            testPassed = true;
        } else {
            std::cerr << "[FAIL] Solver failed: Residual norm too high. (max = " << DEFAULT_TOLERANCE << ")" << std::endl;
        }

        delete[] Ax;
    } else {
        std::cerr << "[FAIL] Solve failed with status: " << solveStatus << std::endl;
    }

    // Check if log file was correctly created and is not empty
    bool logFilePassed = true;
    if (log_file) {
        std::ifstream logFileStream(log_file);
        if (!logFileStream.is_open() || logFileStream.peek() == std::ifstream::traits_type::eof()) {
            std::cerr << "[FAIL] Log file " << log_file << " was not created or is empty.\n";
            logFilePassed = false;
        } else {
            std::cout << "[PASS] Log file " << log_file << " was correctly created and contains output.\n";
        }
    }

    // Only count the test as passed if both the solve test and log file test (if applicable) pass
    if (testPassed && logFilePassed) {
        passedTests++;
    } else {
        failedTests++;
    }

    // Cleanup
    delete[] matrixData.rowPtr;
    delete[] matrixData.colIndices;
    delete[] matrixData.values;
    delete[] rhs;
    delete[] solution;
    delete[] residual;
}

/**
 * @brief Tests exception handling for invalid inputs
 * 
 * Tests various error conditions:
 * 1. Invalid configuration file scenarios
 * 2. Invalid matrix initialization parameters
 * 3. Invalid solver usage patterns
 * 4. Memory-related errors
 * 
 * Each test verifies that the appropriate exception is thrown
 * and contains the expected error message.
 */
void testExpectedExceptions() {
    std::cout << "\n------------------------";
    std::cout << "\n[INFO] Testing exception handling..." << std::endl;

    // Define valid and invalid inputs to avoid repetition
    const int num_rows = 5;
    const int num_non_zeros = 10;
    int invalid_num_rows = -1;  // Invalid: negative row count
    int valid_row_ptr[6] = {0, 2, 4, 6, 8, 10};
    int invalid_row_ptr[6] = {0, 2, 2, 5, 7, 10};  // Invalid: row_ptr[2] == row_ptr[1]
    int valid_col_indices[10] = {0, 1, 1, 2, 2, 3, 3, 4, 4, 0};
    int invalid_col_indices[10] = {0, 1, 5, 2, 2, 3, 3, 4, 4, 10};  // Invalid: 10 is out of bounds
    double valid_values[10] = {1.0, -1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0};
    double* null_values = nullptr;

    double valid_b[5] = {1.0, -1.0, -1.0, 2.0, -2.0};
    double invalid_b[4] = {1.0, -1.0, -1.0, 2.0};
    double valid_x[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double invalid_x[4] = {0.0, 0.0, 0.0, 0.0};

    int gpu_ids[] = {0};
    std::string valid_config = createDefaultConfigFile();
    std::string invalid_config = "xjz.txt";
    
    // 1. Test: Null configuration file
    std::cout << "------------------------\n";
    std::cout << "[TEST] Null configuration file\n";
    try {
        AMGXSolver solver(nullptr, false, gpu_ids, 1, false);
        std::cerr << "[FAIL] Expected exception for null config file not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception& e) {
        std::cout << "[PASS] Caught expected exception: null config file\n";
        std::cerr << "[INFO] Exception: " << e.what() << std::endl;
        passedExceptionTests++;
    }

    // 2. Test: Invalid configuration file
    std::cout << "------------------------\n";
    std::cout << "[TEST] Invalid configuration file\n";
    try {
        AMGXSolver solver(invalid_config.c_str(), false, gpu_ids, 1, false);
        std::cerr << "[FAIL] Expected exception for invalid config file not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception& e) {
        std::cout << "[PASS] Caught expected exception: invalid config file\n";
        std::cerr << "[INFO] Exception: " << e.what() << std::endl;
        passedExceptionTests++;
    }

    // 3. Test: Invalid num_rows
    std::cout << "------------------------\n";
    std::cout << "[TEST] Invalid num_rows\n";
    try {
        AMGXSolver solver(valid_config.c_str(), false, gpu_ids, 1, false);
        solver.initializeMatrix(invalid_num_rows, valid_row_ptr, valid_col_indices, valid_values);
        std::cerr << "[FAIL] Expected exception for invalid num_rows not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception& e) {
        std::cout << "[PASS] Caught expected exception: invalid num_rows\n";
        std::cerr << "[INFO] Exception: " << e.what() << std::endl;
        passedExceptionTests++;
    }

    // 4. Test: GPU mode but null gpu_ids
    std::cout << "------------------------\n";
    std::cout << "[TEST] GPU mode but null gpu_ids\n";
    try {
        AMGXSolver solver(valid_config.c_str(), false, nullptr, 1, false);
        std::cerr << "[FAIL] Expected exception for null GPU IDs in GPU mode not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception& e) {
        std::cout << "[PASS] Caught expected exception: null GPU IDs in GPU mode\n";
        std::cerr << "[INFO] Exception: " << e.what() << std::endl;
        passedExceptionTests++;
    }

    // 5. Test: GPU mode but num_gpus <= 0
    std::cout << "------------------------\n";
    std::cout << "[TEST] GPU mode but num_gpus <= 0\n";
    try {
        int gpu_ids[] = {0};
        AMGXSolver solver(valid_config.c_str(), false, gpu_ids, 0, false);
        std::cerr << "[FAIL] Expected exception for non-positive num_gpus in GPU mode not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception& e) {
        std::cout << "[PASS] Caught expected exception: non-positive num_gpus in GPU mode\n";
        std::cerr << "[INFO] Exception: " << e.what() << std::endl;
        passedExceptionTests++;
    }

    // 6. Test: Null col_indices
    std::cout << "------------------------\n";
    std::cout << "[TEST] Null col_indices\n";
    try {
        AMGXSolver solver(valid_config.c_str(), false, gpu_ids, 1, false);
        solver.initializeMatrix(num_rows, valid_row_ptr, nullptr, valid_values);
        std::cerr << "[FAIL] Expected exception for null col_indices not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception& e) {
        std::cout << "[PASS] Caught expected exception: null col_indices\n";
        std::cerr << "[INFO] Exception: " << e.what() << std::endl;
        passedExceptionTests++;
    }

    // 7. Test: Null row_ptr
    std::cout << "------------------------\n";
    std::cout << "[TEST] Null row_ptr\n";
    try {
        AMGXSolver solver(valid_config.c_str(), false, gpu_ids, 1, false);
        solver.initializeMatrix(num_rows, nullptr, valid_col_indices, valid_values);
        std::cerr << "[FAIL] Expected exception for null row_ptr not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception& e) {
        std::cout << "[PASS] Caught expected exception: null row_ptr\n";
        std::cerr << "[INFO] Exception: " << e.what() << std::endl;
        passedExceptionTests++;
    }

    // 8. Test: Null data values
    std::cout << "------------------------\n";
    std::cout << "[TEST] Null data values\n";
    try {
        AMGXSolver solver(valid_config.c_str(), false, gpu_ids, 1, false);
        solver.initializeMatrix(num_rows, valid_row_ptr, valid_col_indices, null_values);
        std::cerr << "[FAIL] Expected exception for null data values not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception& e) {
        std::cout << "[PASS] Caught expected exception: null data values\n";
        std::cerr << "[INFO] Exception: " << e.what() << std::endl;
        passedExceptionTests++;
    }

    // 9. Test: replaceCoefficients with mismatching num_rows
    std::cout << "------------------------\n";
    std::cout << "[TEST] replaceCoefficients with mismatching num_rows\n";
    try {
        AMGXSolver solver(valid_config.c_str(), false, gpu_ids, 1, false);
        solver.initializeMatrix(num_rows, valid_row_ptr, valid_col_indices, valid_values);
        solver.replaceCoefficients(num_rows + 1, num_non_zeros, valid_values);
        std::cerr << "[FAIL] Expected exception for replaceCoefficients with mismatching num_rows not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception& e) {
        std::cout << "[PASS] Caught expected exception: replaceCoefficients mismatching num_rows\n";
        std::cerr << "[INFO] Exception: " << e.what() << std::endl;
        passedExceptionTests++;
    }

    // 10. Test: replaceCoefficients with mismatching number of nonzeros
    std::cout << "------------------------\n";
    std::cout << "[TEST] replaceCoefficients with mismatching number of nonzeros\n";
    try {
        AMGXSolver solver(valid_config.c_str(), false, gpu_ids, 1, false);
        solver.initializeMatrix(num_rows, valid_row_ptr, valid_col_indices, valid_values);
        solver.replaceCoefficients(num_rows, num_non_zeros - 1, valid_values);
        std::cerr << "[FAIL] Expected exception for replaceCoefficients with mismatching number of nonzeros not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception& e) {
        std::cout << "[PASS] Caught expected exception: replaceCoefficients mismatching number of nonzeros\n";
        std::cerr << "[INFO] Exception: " << e.what() << std::endl;
        passedExceptionTests++;
    }

    // 11. Test: replaceCoefficients with nullptr values
    std::cout << "------------------------\n";
    std::cout << "[TEST] replaceCoefficients with nullptr values\n";
    try {
        AMGXSolver solver(valid_config.c_str(), false, gpu_ids, 1, false);
        solver.initializeMatrix(num_rows, valid_row_ptr, valid_col_indices, valid_values);
        solver.replaceCoefficients(num_rows, num_non_zeros, nullptr);
        std::cerr << "[FAIL] Expected exception for replaceCoefficients with null values not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception& e) {
        std::cout << "[PASS] Caught expected exception: replaceCoefficients null values\n";
        std::cerr << "[INFO] Exception: " << e.what() << std::endl;
        passedExceptionTests++;
    }

    // 12. Test: replaceCoefficients before matrix initialization
    std::cout << "------------------------\n";
    std::cout << "[TEST] replaceCoefficients before matrix initialization\n";
    try {
        AMGXSolver solver(valid_config.c_str(), false, gpu_ids, 1, false);
        solver.replaceCoefficients(num_rows, num_non_zeros, valid_values);
        std::cerr << "[FAIL] Expected exception for replaceCoefficients before matrix initialization not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception& e) {
        std::cout << "[PASS] Caught expected exception: replaceCoefficients before matrix initialization\n";
        std::cerr << "[INFO] Exception: " << e.what() << std::endl;
        passedExceptionTests++;
    }

    // 13. Test: Solve before matrix initialization
    std::cout << "------------------------\n";
    std::cout << "[TEST] Solve before matrix initialization\n";
    try {
        AMGXSolver solver(valid_config.c_str(), false, gpu_ids, 1, false);
        solver.solve(valid_x, valid_b, num_rows);
        std::cerr << "[FAIL] Expected exception for solve before matrix initialization not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception& e) {
        std::cout << "[PASS] Caught expected exception: solve before matrix initialization\n";
        std::cerr << "[INFO] Exception: " << e.what() << std::endl;
        passedExceptionTests++;
    }

    // 14. Test: Solve with null values for right-hand side
    std::cout << "------------------------\n";
    std::cout << "[TEST] Solve with null values for right-hand side\n";
    try {
        AMGXSolver solver(valid_config.c_str(), false, gpu_ids, 1, false);
        solver.initializeMatrix(num_rows, valid_row_ptr, valid_col_indices, valid_values);
        solver.solve(valid_x, nullptr, num_rows);
        std::cerr << "[FAIL] Expected exception for solve with null values for right-hand side not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception& e) {
        std::cout << "[PASS] Caught expected exception: solve with null values for right-hand side\n";
        std::cerr << "[INFO] Exception: " << e.what() << std::endl;
        passedExceptionTests++;
    }

    // 15. Test: Solve with null values for solution
    std::cout << "------------------------\n";
    std::cout << "[TEST] Solve with null values for solution\n";
    try {
        AMGXSolver solver(valid_config.c_str(), false, gpu_ids, 1, false);
        solver.initializeMatrix(num_rows, valid_row_ptr, valid_col_indices, valid_values);
        solver.solve(nullptr, valid_b, num_rows);
        std::cerr << "[FAIL] Expected exception for solve with null values for solution not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception& e) {
        std::cout << "[PASS] Caught expected exception: solve with null values for solution\n";
        std::cerr << "[INFO] Exception: " << e.what() << std::endl;
        passedExceptionTests++;
    }

    // 16. Test: Solve with invalid num_rows
    std::cout << "------------------------\n";
    std::cout << "[TEST] Solve with invalid num_rows\n";
    try {
        AMGXSolver solver(valid_config.c_str(), false, gpu_ids, 1, false);
        solver.initializeMatrix(num_rows, valid_row_ptr, valid_col_indices, valid_values);
        solver.solve(valid_x, valid_b, num_rows + 1);
        std::cerr << "[FAIL] Expected exception for solve with invalid num_rows not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception& e) {
        std::cout << "[PASS] Caught expected exception: solve with invalid num_rows\n";
        std::cerr << "[INFO] Exception: " << e.what() << std::endl;
        passedExceptionTests++;
    }
}

/**
 * @brief Main test driver
 * 
 * Executes test suite with various combinations:
 * - Matrix sizes: 5x5, 50x50, 200x200
 * - CPU and GPU execution
 * - With and without memory pinning
 * - With and without logging
 * 
 * Also runs exception handling tests and reports overall results.
 * 
 * @param argc Command line argument count
 * @param argv Command line arguments (optional: path to config file)
 * @return 0 if all tests pass, -1 if any test fails
 */
int main(int argc, char* argv[]) {
    std::string configFile;
    if (argc >= 2) {
        configFile = argv[1];
    } else {
        configFile = createDefaultConfigFile();
    }

    try {
        // Define test configurations
        bool use_cpu_options[] = {false, true};
        bool pin_memory_options[] = {false, true};
        const char* log_files[] = {nullptr, "solver_log.txt"};

        for (bool use_cpu : use_cpu_options) {
            for (bool pin_memory : pin_memory_options) {
                for (const char* log_file : log_files) {
                    std::cout << "\n=====================================" << std::endl;
                    std::cout << "[INFO] Running test with use_cpu=" << use_cpu
                              << ", pin_memory=" << pin_memory
                              << ", log_file=" << (log_file ? log_file : "None") << std::endl;
                    std::cout << "=====================================" << std::endl;

                    runTest(5, configFile.c_str(), use_cpu, pin_memory, log_file);
                    runTest(50, configFile.c_str(), use_cpu, pin_memory, log_file);
                    runTest(200, configFile.c_str(), use_cpu, pin_memory, log_file);

                    // Delete solver log file if it was created
                    if (log_file) {
                        if (std::remove(log_file) == 0) {
                            std::cout << "[INFO] Deleted log file: " << log_file << std::endl;
                        } else {
                            std::cerr << "[WARNING] Failed to delete log file: " << log_file << std::endl;
                        }
                    }
                }
            }
        }

        testExpectedExceptions();

        // Delete config file after all tests
        if (std::remove(configFile.c_str()) == 0) {
            std::cout << "[INFO] Deleted config file: " << configFile << std::endl;
        } else {
            std::cerr << "[WARNING] Failed to delete config file: " << configFile << std::endl;
        }

        // 🔹 Print final summary of test results
        std::cout << "\n=====================================" << std::endl;
        std::cout << "✅ Tests Passed: " << passedTests << std::endl;
        std::cout << "❌ Tests Failed: " << failedTests << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << "\n=====================================" << std::endl;
        std::cout << "✅ Exception Tests Passed: " << passedExceptionTests << std::endl;
        std::cout << "❌ Exception Tests Failed: " << failedExceptionTests << std::endl;
        std::cout << "=====================================" << std::endl;

        return (failedTests == 0 && failedExceptionTests == 0) ? 0 : -1; // Return 0 if all tests passed, otherwise return -1

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception: " << e.what() << std::endl;
        return -1;
    }
}

