#include <cstdio>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <sstream>
#include <stdexcept>
#include "../solver/AMGXSolver.h"

struct MatrixData {
    int rows;
    int cols;
    int nonZeros;
    int* rowPtr;
    int* colIndices;
    double* values;
};

constexpr double DEFAULT_TOLERANCE = 1e-6;

// Create a temporary config file if none is provided
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


// CSR matrix-vector multiplication: Ax = mat * x
void multiplyCSR(const MatrixData& mat, const double* x, double* Ax) {
    for (int i = 0; i < mat.rows; ++i) {
        Ax[i] = 0.0;
        for (int j = mat.rowPtr[i]; j < mat.rowPtr[i + 1]; ++j) {
            Ax[i] += mat.values[j] * x[mat.colIndices[j]];
        }
    }
}

// Compute residual norm ||r||_2 = sqrt(sum r_i^2)
double calculateResidualNorm(double* r, int size) {
    double norm = 0.0;
    for (int i = 0; i < size; ++i) {
        norm += r[i] * r[i];
    }
    return std::sqrt(norm);
}

// Function to create different test matrices
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

// Global counters for passed/failed tests
int passedTests = 0;
int failedTests = 0;
int passedExceptionTests = 0;
int failedExceptionTests = 0;

// Runs a test on a given matrix size
void runTest(int size, const char* amgx_config_file, bool use_cpu = false, bool pin_memory = true, const char* log_file = nullptr) {
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

// Tests expected exceptions with invalid inputs
void testExpectedExceptions() {
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

    // 1. Test: Null configuration file
    try {
        AMGXSolver solver(nullptr, false, nullptr, 0, false);
        std::cerr << "[FAIL] Expected exception for null config file not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception&) {
        std::cout << "[PASS] Caught expected exception: null config file\n";
        passedExceptionTests++;
    }

    // 2. Test: Invalid num_rows
    try {
        AMGXSolver solver("config.json", false, nullptr, 1, false);
        solver.initializeMatrix(invalid_num_rows, valid_row_ptr, valid_col_indices, valid_values);
        std::cerr << "[FAIL] Expected exception for invalid num_rows not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception&) {
        std::cout << "[PASS] Caught expected exception: invalid num_rows\n";
        passedExceptionTests++;
    }

    // 3. Test: GPU mode but null gpu_ids
    try {
        AMGXSolver solver("config.json", false, nullptr, 1, false);
        std::cerr << "[FAIL] Expected exception for null GPU IDs in GPU mode not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception&) {
        std::cout << "[PASS] Caught expected exception: null GPU IDs in GPU mode\n";
        passedExceptionTests++;
    }

    // 4. Test: GPU mode but num_gpus <= 0
    try {
        int gpu_ids[] = {0};
        AMGXSolver solver("config.json", false, gpu_ids, 0, false);
        std::cerr << "[FAIL] Expected exception for non-positive num_gpus in GPU mode not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception&) {
        std::cout << "[PASS] Caught expected exception: non-positive num_gpus in GPU mode\n";
        passedExceptionTests++;
    }

    // 5. Test: Null col_indices
    try {
        AMGXSolver solver("config.json", false, nullptr, 1, false);
        solver.initializeMatrix(num_rows, valid_row_ptr, nullptr, valid_values);
        std::cerr << "[FAIL] Expected exception for null col_indices not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception&) {
        std::cout << "[PASS] Caught expected exception: null col_indices\n";
        passedExceptionTests++;
    }

    // 6. Test: Invalid col_indices
    try {
        AMGXSolver solver("config.json", false, nullptr, 1, false);
        solver.initializeMatrix(num_rows, valid_row_ptr, invalid_col_indices, valid_values);
        std::cerr << "[FAIL] Expected exception for invalid col_indices not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception&) {
        std::cout << "[PASS] Caught expected exception: invalid col_indices\n";
        passedExceptionTests++;
    }

    // 7. Test: Null row_ptr
    try {
        AMGXSolver solver("config.json", false, nullptr, 1, false);
        solver.initializeMatrix(num_rows, nullptr, valid_col_indices, valid_values);
        std::cerr << "[FAIL] Expected exception for null row_ptr not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception&) {
        std::cout << "[PASS] Caught expected exception: null row_ptr\n";
        passedExceptionTests++;
    }

    // 8. Test: Invalid row_ptr
    try {
        AMGXSolver solver("config.json", false, nullptr, 1, false);
        solver.initializeMatrix(num_rows, invalid_row_ptr, valid_col_indices, valid_values);
        std::cerr << "[FAIL] Expected exception for invalid row_ptr not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception&) {
        std::cout << "[PASS] Caught expected exception: invalid row_ptr\n";
        passedExceptionTests++;
    }

    // 9. Test: Null data values
    try {
        AMGXSolver solver("config.json", false, nullptr, 1, false);
        solver.initializeMatrix(num_rows, valid_row_ptr, valid_col_indices, null_values);
        std::cerr << "[FAIL] Expected exception for null data values not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception&) {
        std::cout << "[PASS] Caught expected exception: null data values\n";
        passedExceptionTests++;
    }

    // 10. Test: replaceCoefficients with mismatching num_rows
    try {
        AMGXSolver solver("config.json", false, nullptr, 1, false);
        solver.initializeMatrix(num_rows, valid_row_ptr, valid_col_indices, valid_values);
        solver.replaceCoefficients(num_rows + 1, num_non_zeros, valid_values);
        std::cerr << "[FAIL] Expected exception for replaceCoefficients with mismatching num_rows not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception&) {
        std::cout << "[PASS] Caught expected exception: replaceCoefficients mismatching num_rows\n";
        passedExceptionTests++;
    }

    // 11. Test: replaceCoefficients with mismatching number of nonzeros
    try {
        AMGXSolver solver("config.json", false, nullptr, 1, false);
        solver.initializeMatrix(num_rows, valid_row_ptr, valid_col_indices, valid_values);
        solver.replaceCoefficients(num_rows, num_non_zeros - 1, valid_values);
        std::cerr << "[FAIL] Expected exception for replaceCoefficients with mismatching number of nonzeros not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception&) {
        std::cout << "[PASS] Caught expected exception: replaceCoefficients mismatching number of nonzeros\n";
        passedExceptionTests++;
    }

    // 12. Test: replaceCoefficients with nullptr values
    try {
        AMGXSolver solver("config.json", false, nullptr, 1, false);
        solver.initializeMatrix(num_rows, valid_row_ptr, valid_col_indices, valid_values);
        solver.replaceCoefficients(num_rows, num_non_zeros, nullptr);
        std::cerr << "[FAIL] Expected exception for replaceCoefficients with null values not thrown.\n";
        failedExceptionTests++;
    } catch (const std::exception&) {
        std::cout << "[PASS] Caught expected exception: replaceCoefficients null values\n";
        passedExceptionTests++;
    }
}

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

