# Final Project Proposal: # AMGX Solver with Python Interface
- Author: Gustavo A. Araújo R.
- Email: garaujor@stanford.edu

## Overview
This project integrates NVIDIA's AMGX solver into a structured testing pipeline with a C++ implementation (`AMGXSolver.cpp`) and a Python wrapper (`pyAMGXSolver`). The goal is to provide an efficient interface for solving large sparse linear systems that can be eventually be merged into the OpenSees finite element framework.

OpenSees (Open System for Earthquake Engineering Simulation) is an open-source software framework written in C++ with a Python interpreter. It is widely used in civil engineering to simulate structural and geotechnical systems under various loading conditions, including earthquakes. The [OpenSees Source Code](https://github.com/OpenSees/OpenSees) is available on GitHub, with its Python Interpreter documented in the [OpenSeesPy Documentation](https://openseespydoc.readthedocs.io/en/latest/).

Efforts to introduce GPU acceleration into OpenSees have been made in the past, such as the integration of the Cusp library ([OpenSees Cusp Integration](https://opensees.berkeley.edu/wiki/index.php/Cusp)). However, Cusp has since become obsolete, and other libraries that leverage modern GPU hardware provide new opportunities for GPU acceleration. For this project, I plan to leverage **AmgX**, a library designed specifically for solving large sparse linear systems with algebraic multigrid methods.

By integrating this modern libraries, the idea is to bring GPU-based solvers back to OpenSees.

## Objectives
1. Implement GPU-accelerated solvers that mimic the interface of the sparse solvers used in OpenSees using AmgX.
2. Benchmark the performance of GPU-based solvers for a set of structural models of varying size and complexity selected from the SuiteSparse data set.

The provided `main.py` script automates the process of downloading SuiteSparse matrices, running AMGX with multiple configurations, and analyzing the results.

## Features
- **C++ Core:** The `AMGXSolver.cpp` class interfaces with AMGX to perform iterative and direct sparse solvers. This is the interface intended to be merged into OpenSees
- **Python Wrapper:** `pyAMGXSolver` exposes the C++ solver to Python using `pybind11`, allowing easy testing.
- **Automated Testing:** The `main.py` script runs AMGX on multiple matrices, collects convergence metrics, and generates plots.
- **Flexible Configuration:** Supports different AMGX solver configurations via JSON files.
- **CSV Logging:** Saves performance results (solve time, iteration count, and solver status) for analysis.

---

## Compilation and Installation

### 1️⃣ **Clone the Repository**
Start by cloning this repository:

```bash
git clone https://github.com/gaaraujo/BIODS253-Final-Project.git
cd BIODS253-Final-Project
```

---

### 2️⃣ **Python Environment Setup**
Create a virtual environment to ensure compatibility:

```bash
python3 -m venv amgx_env
source amgx_env/bin/activate  # macOS/Linux
# On Windows: amgx_env\Scripts\activate

pip install -r requirements.txt
```

This will install all required Python dependencies.

---

### 3️⃣ **Installing AMGX**
This project relies on **AMGX**, NVIDIA’s algebraic multigrid solver. You need to install it before compiling the solver. 
Follow the installation instructions provided in the [AMGX GitHub repository](https://github.com/NVIDIA/AMGX).

---

### 4️⃣ **C++ Compilation and Setup**
Ensure the following dependencies are installed:
- **CMake**
- **GCC or Clang** (C++ compiler)
- **CUDA Toolkit** (if using GPU acceleration)
- **AMGX** (installed in step 3)

Compile the solver:

```bash
mkdir build
cd build
cmake .. -DAMGX_CUSTOM_PATH=/path/to/amgx
make -j$(nproc)
```

To verify compilation:

```bash
./test_solver
```

Expected output:

```
=====================================
✅ Tests Passed: 24
❌ Tests Failed: 0
=====================================

=====================================
✅ Exception Tests Passed: 16
❌ Exception Tests Failed: 0
=====================================
```

---

### 5️⃣ **Verifying the Python Wrapper**
Ensure the Python wrapper is functioning correctly by running:

```bash
python3 SRC/tests/test_pyAMGXSolver.py
```

Expected output:

```
=====================================
Exception Tests Summary:
✅ Passed: 9, ❌ Failed: 0
=====================================

=====================================
✅ Solver Tests Passed: 24
❌ Solver Tests Failed: 0
=====================================
```

This confirms that the solver and Python wrapper are correctly installed and working. 🚀

---

## Using the Python Wrapper

### Importing the Solver
```python
import pyAMGXSolver
solver = pyAMGXSolver.AMGXSolver("config.json", use_cpu=False, gpu_ids=[0])
```

### Initializing the Matrix
```python
import numpy as np
row_ptr = np.array([0, 2, 4], dtype=np.int32)
col_indices = np.array([0, 1, 1, 2], dtype=np.int32)
values = np.array([4.0, -1.0, -1.0, 4.0], dtype=np.float64)
solver.initialize_matrix(row_ptr, col_indices, values)
```

### Solving a System
```python
b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
x, status, iterations, residual = solver.solve(b)
print(f"Solution: {x}, Status: {status}, Iterations: {iterations}, Residual: {residual}")
```

### Cleaning Up
```python
solver.cleanup()
```

---

## Running the Main Script
The `main.py` script automates downloading matrices, running AMGX, and collecting results.

### Command-Line Arguments
The script supports the following optional arguments:
```sh
python3 main.py -MATRIX_DIR <path> -LOG_DIR <path> -CONFIG_DIR <path> -PLOT_DIR <path> -TEMP_DIR <path> -INPUT_CSV_FILE <file> -OUTPUT_CSV_FILE <file>
```

### Example Usage
To run with default settings:
```sh
python3 main.py
```
To specify custom directories:
```sh
python3 main.py -MATRIX_DIR /custom/matrices -LOG_DIR /custom/logs
```

---

## Output and Logs
- **Terminal Output:**
  - Displays matrix name, configuration used, solver status, and performance metrics.
- **CSV File (`matrix_test_results.csv`):**
  - Stores results including solve time, iteration count, and convergence status.
- **Log Files (`logs/*.log`):**
  - Contains AMGX output for debugging purposes.
- **Plots (`plots/`):**
  - Generates performance visualizations.

### Understanding Solver Status
- `Solver Status = 0` → Successful solution.
- `Solver Status < 0` → Failure (divergence or numerical issues).
- **Note:** If `DENSE_LU_SOLVER` is used, the residual reported by AMGX is incorrect as it does not compute the final residual.

### Notes on AMGX Configuration Files
It is important that the implemented Python wrapper requires configuration files where "monitor_residual" = 1 and "store_res_history" = 1 to work correctly.
The core C++ interface does not mandatorily require it but it's recommended if residuals want to be tracked without the need for log files.

---

## Summary
This project provides a robust framework for testing AMGX solvers with structural matrices. It includes:
- A **C++ core solver** for AMGX integration.
- A **Python wrapper** for testing and usability.
- A **benchmarking script (`main.py`)** for automated testing and analysis.

