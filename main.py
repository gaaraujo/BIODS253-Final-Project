import os
import time
import numpy as np
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pyAMGXSolver
from amgx_log_parser import parse_amgx_log  # Function to extract performance data

# Directory paths
MATRIX_DIR = "matrix_tests/matrices"
LOG_DIR = "matrix_tests/logs"
CONFIG_DIR = "matrix_tests/configs"
PLOT_DIR = "matrix_tests/plots"

# Ensure necessary directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def load_matrix(mtx_file):
    """Load a matrix in Matrix Market (.mtx) format and return as CSR format."""
    print(f"[INFO] Loading matrix: {mtx_file}")
    
    A = scipy.io.mmread(mtx_file)  # Read matrix from file
    if not sp.issparse(A):
        raise ValueError("Loaded matrix is not sparse.")
    
    A_csr = A.tocsr()  # Convert to CSR format (needed for AMGX)
    return A_csr

def run_test(matrix_path, config_file, use_cpu=False, pin_memory=True):
    """Runs AMGX solver on a given matrix and logs results."""
    matrix_name = os.path.basename(matrix_path).replace(".mtx", "")
    config_name = os.path.basename(config_file).replace(".json", "")
    log_file = os.path.join(LOG_DIR, f"{matrix_name}_{config_name}.log")

    # **Clear the log file before running**
    open(log_file, "w").close()

    try:
        A = load_matrix(matrix_path)  # Load the sparse matrix
        num_rows = A.shape[0]

        # Generate a right-hand side vector b
        b = np.ones(num_rows, dtype=np.float64)

        # Convert to CSR format
        row_ptr, col_indices, values = A.indptr, A.indices, A.data

        # Initialize solver
        solver = pyAMGXSolver.AMGXSolver(config_file, use_cpu=use_cpu, gpu_ids=[0], pin_memory=pin_memory, log_file=log_file)
        solver.initialize_matrix(row_ptr, col_indices, values)

        # Solve Ax = b
        start_time = time.time()
        x = solver.solve(b)
        end_time = time.time()
        solver.cleanup()

        # Compute and log performance metrics
        elapsed_time = end_time - start_time
        log_data = parse_amgx_log(log_file)

        solver_status = log_data.get("solver_status", None)
        amgx_time = log_data.get("total_time", None)
        iterations = log_data.get("total_iterations", None)
        final_residual = log_data.get("final_residual", None)

        print(f"[RESULT] {matrix_name} ({config_name}): Num. Rows={num_rows}, Solver Status={solver_status}, Residual={final_residual}, Iterations={iterations}, Elapsed Time={elapsed_time:.6f} s, AMGX Time={amgx_time:.6f} s")

        return num_rows, elapsed_time, log_data, config_name

    except Exception as e:
        print(f"[ERROR] Failed to solve {matrix_name} ({config_name}): {e}")
        return None

def main():
    """Main entry point: iterates through matrices and config files, and generates plots."""
    matrix_files = sorted([f for f in os.listdir(MATRIX_DIR) if f.endswith(".mtx")])
    config_files = sorted([os.path.join(CONFIG_DIR, f) for f in os.listdir(CONFIG_DIR) if f.endswith(".json")])

    if not matrix_files:
        print("[ERROR] No Matrix Market (.mtx) files found.")
        return
    if not config_files:
        print("[ERROR] No configuration files found in 'configs/'.")
        return

    print(f"[INFO] Found {len(matrix_files)} matrices and {len(config_files)} config files. Running tests...\n")

    results = []

    for matrix_file in matrix_files:
        matrix_path = os.path.join(MATRIX_DIR, matrix_file)
        for config_file in config_files:
            result = run_test(matrix_path, config_file)
            if result:
                results.append(result)

    # Generate plots
    plot_results(results)


def plot_results(results):
    """Plots time and iterations against matrix sizes, marking failed solves with an 'X'."""
    import matplotlib.pyplot as plt

    # Dictionary to store results for plotting
    data = {}

    for num_rows, elapsed_time, log_data, config_name in results:
        amgx_time = log_data.get("total_time", None)
        iterations = log_data.get("total_iterations", None)
        solver_status = log_data.get("solver_status", None)

        if config_name not in data:
            data[config_name] = {
                "sizes": [], "elapsed_times": [], "amgx_times": [], "iterations": [],
                "failed_sizes": [], "failed_amgx_times": [], "failed_iterations": []
            }

        if solver_status == 0: 
            data[config_name]["sizes"].append(num_rows)
            data[config_name]["elapsed_times"].append(elapsed_time)
            data[config_name]["amgx_times"].append(amgx_time)
            data[config_name]["iterations"].append(iterations)
        else:
            data[config_name]["failed_sizes"].append(num_rows)
            data[config_name]["failed_amgx_times"].append(amgx_time)
            data[config_name]["failed_iterations"].append(iterations)

    # Plot AMGX time vs matrix size
    plt.figure(figsize=(10, 6))
    for config_name, values in data.items():
        # Plot successful cases
        line, = plt.plot(values["sizes"], values["amgx_times"], marker='o', linestyle='-', label=config_name)

        # Use the same color for failed cases
        plt.scatter(values["failed_sizes"], values["failed_amgx_times"], marker='x', color=line.get_color(), label=f"{config_name} (Failed)")


    plt.xlabel("Matrix Size (Number of Rows)")
    plt.ylabel("AMGX Solve Time (s)")
    plt.title("AMGX Solve Time vs Matrix Size")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "amgx_solve_time.png"))
    plt.show()

    # Plot iterations vs matrix size
    plt.figure(figsize=(10, 6))
    for config_name, values in data.items():
        plt.plot(values["sizes"], values["iterations"], marker='s', linestyle='-', label=config_name)

        # Mark failed cases with an "X"
        plt.scatter(values["failed_sizes"], values["failed_iterations"], marker='x', color='red', label=f"{config_name} (Failed)")

    plt.xlabel("Matrix Size (Number of Rows)")
    plt.ylabel("Number of Iterations")
    plt.title("Iterations vs Matrix Size")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "iterations_vs_matrix_size.png"))
    plt.show()

if __name__ == "__main__":
    main()
