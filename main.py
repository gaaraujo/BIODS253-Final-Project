import os
import time
import numpy as np
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt
import urllib.request
import tarfile
import shutil
import pandas as pd
import pyAMGXSolver
from amgx_log_parser import parse_amgx_log  # Function to extract performance data

import argparse

def parse_arguments():
    """Parse command-line arguments for directory paths.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            - MATRIX_DIR: Directory containing matrix files (.mtx)
            - LOG_DIR: Directory to store solver logs
            - CONFIG_DIR: Directory containing solver config files (.json)
            - PLOT_DIR: Directory to save plots
            - TEMP_DIR: Temporary directory for downloads
            - INPUT_CSV_FILE: CSV file containing matrix metadata
            - OUTPUT_CSV_FILE: CSV file containing test results
    """
    parser = argparse.ArgumentParser(description="Run AMGX solver tests on a set of matrices from the SuiteSparse data set with different AMGX configuration files.")
    
    parser.add_argument("-MATRIX_DIR", type=str, default="matrix_tests/matrices",
                        help="Directory containing matrix files (.mtx)")
    parser.add_argument("-LOG_DIR", type=str, default="matrix_tests/logs",
                        help="Directory to store solver logs")
    parser.add_argument("-CONFIG_DIR", type=str, default="matrix_tests/configs",
                        help="Directory containing solver config files (.json)")
    parser.add_argument("-PLOT_DIR", type=str, default="matrix_tests/plots",
                        help="Directory to save plots")
    parser.add_argument("-TEMP_DIR", type=str, default="matrix_tests/temp",
                        help="Temporary directory for downloads")
    parser.add_argument("-INPUT_CSV_FILE", type=str, default="matrix_tests/matrices.csv",
                        help="CSV file containing matrix metadata")
    parser.add_argument("-OUTPUT_CSV_FILE", type=str, default="matrix_tests/plots/matrix_test_results.csv",
                        help="CSV file containing timings, number of iterations, and convergence")

    return parser.parse_args()

# Base URL for downloading SuiteSparse matrices
BASE_URL = "https://suitesparse-collection-website.herokuapp.com/MM"

def download_and_extract(group, matrix_name):
    """Download and extract a SuiteSparse matrix in .mtx format if it's missing.

    Downloads matrix from SuiteSparse collection, extracts it, and saves in the
    specified matrix directory. Cleans up temporary files after extraction.

    Args:
        group (str): Matrix group name in SuiteSparse collection
        matrix_name (str): Name of the matrix to download

    Note:
        Uses global TEMP_DIR and MATRIX_DIR for file operations
    """
    file_url = f"{BASE_URL}/{group}/{matrix_name}.tar.gz"
    archive_path = os.path.join(TEMP_DIR, f"{matrix_name}.tar.gz")
    extract_path = os.path.join(TEMP_DIR, matrix_name)
    final_mtx_path = os.path.join(MATRIX_DIR, f"{matrix_name}.mtx")

    # Check if matrix already exists
    if os.path.exists(final_mtx_path):
        print(f"✅ {matrix_name}.mtx already exists, skipping download.")
        return

    try:
        print(f"📥 Downloading {matrix_name} from {file_url}...")
        urllib.request.urlretrieve(file_url, archive_path)

        print(f"📦 Extracting {archive_path}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_path)

        # Find the .mtx file inside the extracted folder
        for root, _, files in os.walk(extract_path):
            for file in files:
                if file == f"{matrix_name}.mtx":
                    extracted_mtx = os.path.join(root, file)
                    shutil.move(extracted_mtx, final_mtx_path)
                    print(f"✅ Saved {final_mtx_path}")

        # Clean up
        os.remove(archive_path)  # Remove the tar.gz file
        shutil.rmtree(extract_path)  # Remove extracted folder
        print(f"🧹 Cleaned up temporary files for {matrix_name}")

    except Exception as e:
        print(f"[ERROR] Failed to process {matrix_name}: {e}")

def load_matrix(mtx_file):
    """Load a matrix in Matrix Market (.mtx) format and return as CSR format.

    Args:
        mtx_file (str): Path to the .mtx file to load

    Returns:
        scipy.sparse.csr_matrix: Matrix in CSR format

    Raises:
        ValueError: If loaded matrix is not sparse
    """
    print(f"[INFO] Loading matrix: {mtx_file}")
    
    A = scipy.io.mmread(mtx_file)  # Read matrix from file
    if not sp.issparse(A):
        raise ValueError("Loaded matrix is not sparse.")
    
    A_csr = A.tocsr()  # Convert to CSR format (needed for AMGX)
    return A_csr

def run_test(matrix_path, config_file, use_cpu=False, pin_memory=True, k=11):
    """Run AMGX solver tests on a given matrix with specified configuration.

    Performs k solver runs, discarding the first run to account for GPU warm-up effects.
    If solver fails (status < 0), only the first run is performed.

    Args:
        matrix_path (str): Path to the matrix file (.mtx)
        config_file (str): Path to AMGX configuration file (.json)
        use_cpu (bool, optional): Whether to use CPU instead of GPU. Defaults to False.
        pin_memory (bool, optional): Whether to pin memory. Defaults to True.
        k (int, optional): Number of runs to perform. Defaults to 11.

    Returns:
        tuple: Contains (num_rows, avg_elapsed_time, avg_amgx_time, avg_iterations, 
               solver_status, config_name) if successful, None if failed

    Note:
        Uses global LOG_DIR for storing solver logs
    """
    matrix_name = os.path.basename(matrix_path).replace(".mtx", "")
    config_name = os.path.basename(config_file).replace(".json", "")
    log_file = os.path.join(LOG_DIR, f"{matrix_name}_{config_name}.log")

    print("---------------------------------------------")
    try:
        A = load_matrix(matrix_path)  # Load the sparse matrix
        num_rows = A.shape[0]

        print(f"[TEST] {matrix_name} ({config_name}): Num. Rows={num_rows}")

        # Generate a right-hand side vector b
        rng = np.random.default_rng(42)  # Use default generator with fixed seed
        b = rng.random(num_rows, dtype=np.float64)  # Generate random values

        # Convert to CSR format
        row_ptr, col_indices, values = A.indptr, A.indices, A.data

        elapsed_times = []
        amgx_times = []
        amgx_iterations = []
        solver_statuses = []

        # **Clear the log file before each run**
        open(log_file, "w").close()

        # **Create the solver once**
        solver = pyAMGXSolver.AMGXSolver(config_file, use_cpu=use_cpu, gpu_ids=[0], pin_memory=pin_memory, log_file=log_file)

        # **Reinitialize matrix** instead of creating a new solver
        solver.initialize_matrix(row_ptr, col_indices, values)

        # **First run (always executed)**
        start_time = time.time()
        x, status, iterations, residual = solver.solve(b)
        end_time = time.time()

        # Compute and log performance metrics
        elapsed_time = end_time - start_time
        log_data = parse_amgx_log(log_file)

        solver_status = log_data.get("solver_status", None)
        amgx_residual = log_data.get("final_residual", None)
        amgx_time = log_data.get("total_time", None)
        amgx_total_iter = log_data.get("total_iterations", None)

        solver_statuses.append(solver_status)

        # **If solver status is negative, do NOT repeat runs**
        if solver_status < 0:
            avg_elapsed_time = elapsed_time
            avg_amgx_time = amgx_time
            avg_iterations = amgx_total_iter
        else:
            # **Repeat for (k-1) additional runs**
            for i in range(k - 1):
                start_time = time.time()
                x, status, iterations, residual = solver.solve(b)
                end_time = time.time()

                # Compute performance metrics
                elapsed_time = end_time - start_time
                log_data = parse_amgx_log(log_file)

                amgx_time = log_data.get("total_time", None)
                amgx_total_iter = log_data.get("total_iterations", None)

                solver_statuses.append(status)
                elapsed_times.append(elapsed_time)
                amgx_times.append(amgx_time)
                amgx_iterations.append(amgx_total_iter)

            # Compute averages
            avg_elapsed_time = np.mean(elapsed_times) if elapsed_times else None
            avg_amgx_time = np.mean(amgx_times) if amgx_times else None
            avg_iterations = np.mean(amgx_iterations) if amgx_iterations else None

        # Clean up only once
        solver.cleanup()

        solver_status = solver_statuses[-1]
        print(f"[RESULT] {matrix_name} ({config_name}): Num. Rows={num_rows}, Solver Status={solver_status}, "
              f"Avg Residual={amgx_residual}, Avg Iterations={avg_iterations}, "
              f"Avg Elapsed Time={avg_elapsed_time:.6f} s, Avg AMGX Time={avg_amgx_time:.6f} s")

        print("---------------------------------------------")
        return num_rows, avg_elapsed_time, avg_amgx_time, avg_iterations, solver_status, config_name

    except Exception as e:
        print(f"[ERROR] Failed to solve {matrix_name} ({config_name}): {e}")
        return None

def main():
    """Main entry point for the AMGX solver testing pipeline.

    Workflow:
    1. Parse command-line arguments for directory paths
    2. Download missing matrices from SuiteSparse collection
    3. Run solver tests with different configurations
    4. Save results to CSV file
    5. Generate performance plots

    The script processes each matrix with all available configurations,
    tracking solve time, iterations, and convergence status.

    Usage:
        Run 'python main.py --help' to see all available command-line arguments
        and their default values.
    """
    args = parse_arguments()

    global MATRIX_DIR, LOG_DIR, CONFIG_DIR, PLOT_DIR, TEMP_DIR, INPUT_CSV_FILE, OUTPUT_CSV_FILE
    MATRIX_DIR, LOG_DIR, CONFIG_DIR, PLOT_DIR, TEMP_DIR, INPUT_CSV_FILE, OUTPUT_CSV_FILE = (
        args.MATRIX_DIR, args.LOG_DIR, args.CONFIG_DIR, args.PLOT_DIR, args.TEMP_DIR, args.INPUT_CSV_FILE, args.OUTPUT_CSV_FILE
    )

    # Print out the paths being used
    print("\n🔧 Using the following paths:")
    print(f"   📂 MATRIX_DIR  : {MATRIX_DIR}")
    print(f"   📂 LOG_DIR     : {LOG_DIR}")
    print(f"   📂 CONFIG_DIR  : {CONFIG_DIR}")
    print(f"   📂 PLOT_DIR    : {PLOT_DIR}")
    print(f"   📂 TEMP_DIR    : {TEMP_DIR}")
    print(f"   📄 INPUT_CSV_FILE    : {INPUT_CSV_FILE}")
    print(f"   📄 OUTPUT_CSV_FILE    : {OUTPUT_CSV_FILE}\n")
    

    # Ensure necessary directories exist
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(MATRIX_DIR, exist_ok=True)


    df = pd.read_csv(INPUT_CSV_FILE)
    
    # Download missing matrices
    for _, row in df.iterrows():
        matrix_name = row["Name"]
        group = row["Group"]
        
        if not pd.isna(matrix_name) and not pd.isna(group):
            download_and_extract(group, matrix_name)

    # Get list of available matrices
    matrix_files = sorted([f for f in os.listdir(MATRIX_DIR) if f.endswith(".mtx")])
    config_files = sorted([os.path.join(CONFIG_DIR, f) for f in os.listdir(CONFIG_DIR) if f.endswith(".json")])

    if not matrix_files:
        print("[ERROR] No Matrix Market (.mtx) files found.")
        return
    if not config_files:
        print(f"[ERROR] No configuration files found in {CONFIG_DIR}.")
        return

    print(f"[INFO] Found {len(matrix_files)} matrices and {len(config_files)} config files. Running tests...\n")

    results = []
    
    # Dictionary to store results for CSV
    results_dict = {col: [] for col in df.columns}  # Copy original metadata

    # Add new columns for each config file
    for config_file in config_files:
        config_name = os.path.basename(config_file).replace(".json", "")
        results_dict[f"SolveTime_{config_name}"] = []
        results_dict[f"Iterations_{config_name}"] = []
        results_dict[f"Success_{config_name}"] = []

    # Iterate through matrices and test each with all configs
    for idx, matrix_file in enumerate(matrix_files, 1):
        matrix_path = os.path.join(MATRIX_DIR, matrix_file)
        matrix_name = matrix_file.replace(".mtx", "")
        
        print(f"🔄 Processing matrix {idx} / {len(matrix_files)}: {matrix_name}")

        # Find row in original CSV for this matrix
        row_index = df[df["Name"] == matrix_name].index
        if row_index.empty:
            print(f"⚠️ Skipping {matrix_name}: Not found in CSV.")
            continue

        row_index = row_index[0]

        # Store original metadata in results_dict
        for col in df.columns:
            results_dict[col].append(df.at[row_index, col])

        for config_file in config_files:
            config_name = os.path.basename(config_file).replace(".json", "")
            result = run_test(matrix_path, config_file)
            
            if result:
                num_rows, avg_elapsed_time, avg_amgx_time, avg_iterations, solver_status, _ = result
                results.append(result)

                # Store results
                results_dict[f"SolveTime_{config_name}"].append(avg_amgx_time)
                results_dict[f"Iterations_{config_name}"].append(avg_iterations)
                results_dict[f"Success_{config_name}"].append(1 if solver_status == 0 else 0)
            else:
                # Store NaN for failed tests
                results_dict[f"SolveTime_{config_name}"].append(np.nan)
                results_dict[f"Iterations_{config_name}"].append(np.nan)
                results_dict[f"Success_{config_name}"].append(0)

    # Convert dictionary to DataFrame and save
    results_df = pd.DataFrame(results_dict)
    results_csv_path = OUTPUT_CSV_FILE
    results_df.to_csv(results_csv_path, index=False)

    print(f"\n✅ Test results saved to {results_csv_path}")

    # Generate plots
    plot_results(results)


def plot_results(results):
    """Plot performance metrics against matrix sizes.

    Creates two plots:
    1. AMGX solve time vs matrix size
    2. Number of iterations vs matrix size
    
    Successful solves are marked with dots/squares and connected by lines,
    while failed solves are marked with 'X' markers.

    Args:
        results (list): List of tuples containing:
            (num_rows, elapsed_time, amgx_time, amgx_iterations, solver_status, config_name)

    Note:
        Uses global PLOT_DIR to save generated plots
    """
    import matplotlib.pyplot as plt

    # Dictionary to store results for plotting
    data = {}

    for num_rows, elapsed_time, amgx_time, amgx_iterations, solver_status, config_name in results:

        if config_name not in data:
            data[config_name] = {
                "sizes": [], "elapsed_times": [], "amgx_times": [], "iterations": [],
                "failed_sizes": [], "failed_amgx_times": [], "failed_iterations": []
            }

        if solver_status == 0:
            data[config_name]["sizes"].append(num_rows)
            data[config_name]["elapsed_times"].append(elapsed_time)
            data[config_name]["amgx_times"].append(amgx_time)
            data[config_name]["iterations"].append(amgx_iterations)
        else:
            data[config_name]["failed_sizes"].append(num_rows)
            data[config_name]["failed_amgx_times"].append(amgx_time)
            data[config_name]["failed_iterations"].append(amgx_iterations)

    # Plot AMGX time vs matrix size
    plt.figure(figsize=(10, 6))
    for config_name, values in data.items():
        # Sort successful cases by increasing matrix size
        sorted_indices = np.argsort(values["sizes"])
        sorted_sizes = np.array(values["sizes"])[sorted_indices]
        sorted_amgx_times = np.array(values["amgx_times"])[sorted_indices]

        # Sort failed cases by increasing matrix size
        sorted_fail_indices = np.argsort(values["failed_sizes"])
        sorted_failed_sizes = np.array(values["failed_sizes"])[sorted_fail_indices]
        sorted_failed_amgx_times = np.array(values["failed_amgx_times"])[sorted_fail_indices]

        # Plot successful cases
        line, = plt.plot(sorted_sizes, sorted_amgx_times, marker='o', linestyle='-', label=config_name)

        # Use the same color for failed cases
        plt.scatter(sorted_failed_sizes, sorted_failed_amgx_times, marker='x', color=line.get_color(), label=f"{config_name} (Failed)")

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
        # Sort successful cases by increasing matrix size
        sorted_indices = np.argsort(values["sizes"])
        sorted_sizes = np.array(values["sizes"])[sorted_indices]
        sorted_iterations = np.array(values["iterations"])[sorted_indices]

        # Sort failed cases by increasing matrix size
        sorted_fail_indices = np.argsort(values["failed_sizes"])
        sorted_failed_sizes = np.array(values["failed_sizes"])[sorted_fail_indices]
        sorted_failed_iterations = np.array(values["failed_iterations"])[sorted_fail_indices]

        # Plot successful cases
        line, = plt.plot(sorted_sizes, sorted_iterations, marker='s', linestyle='-', label=config_name)

        # Mark failed cases with an "X"
        plt.scatter(sorted_failed_sizes, sorted_failed_iterations, marker='x', color=line.get_color(), label=f"{config_name} (Failed)")

    plt.xlabel("Matrix Size (Number of Rows)")
    plt.ylabel("Number of Iterations")
    plt.title("Iterations vs Matrix Size")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "iterations_vs_matrix_size.png"))
    plt.show()


if __name__ == "__main__":
    main()
