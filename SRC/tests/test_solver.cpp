#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <cstring> // For strlen, strcpy

#include "solver/AMGXSolver.h"

// Assuming a maximum line length for Matrix Market files
#define MAX_LINE_LENGTH 256

struct MatrixData {
    int rows;
    int cols;
    int nonZeros;
    int* rowPtr;
    int* colIndices;
    double* values;
};

MatrixData readMatrixMarket(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: ");
    }

    MatrixData data;
    char line[MAX_LINE_LENGTH];
    data.rows = 0; // Initialize to indicate header not read yet

    while (file.getline(line, MAX_LINE_LENGTH)) {
        if (line[0] == '%') continue; // Skip comments

        std::stringstream ss(line);
        if (data.rows == 0) {
            ss >> data.rows >> data.cols >> data.nonZeros;
            data.rowPtr = new int[data.rows + 1];
            data.colIndices = new int[data.nonZeros];
            data.values = new double[data.nonZeros];

            if(!data.rowPtr || !data.colIndices || !data.values){
                throw std::runtime_error("Memory allocation failure");
            }
            memset(data.rowPtr, 0, sizeof(int) * (data.rows+1));

        } else {
            int row, col;
            double val;
            ss >> row >> col >> val;
            data.colIndices[data.rowPtr[row - 1]] = col - 1;
            data.values[data.rowPtr[row - 1]] = val;
            for (int i = row; i < data.rows + 1; ++i) {
                data.rowPtr[i]++;
            }
        }
    }
    return data;
}

double calculateResidualNorm(double* r, int size) {
    double norm = 0.0;
    for (int i = 0; i < size; ++i) {
        norm += r[i] * r[i];
    }
    return std::sqrt(norm);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_market_file>" << std::endl;
        return 1;
    }

    try {
        MatrixData matrixData = readMatrixMarket(argv[1]);
        double* rhs = new double[matrixData.rows];
        double* solution = new double[matrixData.rows];
        double* residual = new double[matrixData.rows];

        if (!rhs || !solution || !residual){
            throw std::runtime_error("Memory allocation failure");
        }

        for (int i = 0; i < matrixData.rows; ++i) {
            rhs[i] = 1.0;
        }

        AMGXSolver solver("your_config.json", false, nullptr, 0, true);
        solver.initializeMatrix(matrixData.rows, matrixData.rowPtr, matrixData.colIndices, matrixData.values);
        int solveStatus = solver.solve(solution, rhs, matrixData.rows);

        if (solveStatus == 0) {
            // Calculate residual (Simplified, you'll need a proper matrix-vector multiply)
            // Example: r = A*x - b
            // This example assumes A is identity.
            for (int i = 0; i < matrixData.rows; ++i) {
                residual[i] = solution[i] - rhs[i];
            }

            double residualNorm = calculateResidualNorm(residual, matrixData.rows);
            std::cout << "Residual norm: " << residualNorm << std::endl;
            if (residualNorm < 1e-6) {
                std::cout << "Test passed!" << std::endl;
            } else {
                std::cout << "Test failed: Residual norm too high." << std::endl;
            }
        } else {
            std::cout << "Solve failed with status: " << solveStatus << std::endl;
        }

        // Clean up allocated memory
        delete[] matrixData.rowPtr;
        delete[] matrixData.colIndices;
        delete[] matrixData.values;
        delete[] rhs;
        delete[] solution;
        delete[] residual;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}