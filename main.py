import os
import time
import numpy as np
from utils import (
    load_matrix, get_matrix_files,
    save_results_to_csv, plot_results, import_pyAMGXSolver, get_cpu_specs, get_gpu_specs, get_system
)
from amgx_log_parser import parse_amgx_log

# Import pyAMGXSolver after path setup
pyAMGXSolver = import_pyAMGXSolver()

import argparse

def parse_arguments():
    """Parse command-line arguments for directory paths and solver configuration.
    
    Matrix input modes:
    1. With --input_csv: Downloads matrices specified in CSV to matrices_dir
    2. Without --input_csv: Uses existing .mtx files from matrices_dir
    3. Single matrix: Uses specified matrix file
    
    Config input modes:
    1. Directory (default): Uses all configs from config_dir
    2. Single config: Uses specified config file
    
    Default directories:
    - matrices_dir: matrix_tests/matrices
    - config_dir: matrix_tests/configs
    """
    parser = argparse.ArgumentParser(description="Run AMGX solver tests on matrices with different configurations.")
    
    # Matrix inputs
    matrix_group = parser.add_mutually_exclusive_group()
    matrix_group.add_argument("--input_csv", type=str,
                            help="CSV file containing matrix metadata for downloading matrices")
    matrix_group.add_argument("--matrix_file", type=str,
                            help="Path to single matrix file to test")

    # Config inputs
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument("--config_dir", type=str, default="matrix_tests/configs",
                            help="Directory containing solver config files (.json) [default: matrix_tests/configs]")
    config_group.add_argument("--config_file", type=str,
                            help="Path to single config file")

    # Directories
    parser.add_argument("--matrices_dir", type=str, default="matrix_tests/matrices",
                        help="Directory containing/storing matrix files (.mtx) [default: matrix_tests/matrices]")
    parser.add_argument("--log_dir", type=str, default="matrix_tests/logs",
                        help="Directory to store solver logs")
    parser.add_argument("--output_dir", type=str, default="matrix_tests/output",
                        help="Directory to save results and plots")
    parser.add_argument("--output_csv", type=str, default="amgx_results.csv",
                        help="Name of the CSV file to save in output_dir")

    # Solver configuration
    parser.add_argument("--use_cpu", action="store_true", default=False,
                        help="Use CPU instead of GPU for solving")
    parser.add_argument("--no_pin_memory", action='store_true',
                       help="Disable pinned memory for GPU transfers")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of runs for computing average performance")

    args = parser.parse_args()

    # Validate output_csv is just a filename, not a path
    if os.path.sep in args.output_csv:
        parser.error("--output_csv should be a filename, not a path. It will be saved in output_dir.")

    # Create necessary directories
    os.makedirs(args.matrices_dir, exist_ok=True)
    os.makedirs(args.config_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Print configuration
    print("\n🔧 Running with the following configuration:")
    print("\nInput Sources:")
    if args.input_csv:
        print(f"  📥 Downloading matrices from CSV: {args.input_csv}")
    elif args.matrix_file:
        print(f"  📄 Using single matrix file: {args.matrix_file}")
    else:
        print(f"  📂 Using matrices from directory: {args.matrices_dir}")

    if args.config_file:
        print(f"  ⚙️  Using single config file: {args.config_file}")
    else:
        print(f"  ⚙️  Using configs from directory: {args.config_dir}")

    print("\nOutput Locations:")
    print(f"  📂 Output directory: {args.output_dir}")
    print(f"  📊 Results CSV: {os.path.join(args.output_dir, args.output_csv)}")
    print(f"  📝 Log directory: {args.log_dir}")

    print("\nSolver Configuration:")
    print(f"  🖥️  Using {'CPU' if args.use_cpu else 'GPU'}")
    print(f"  🔒 Memory pinning: {'disabled' if args.no_pin_memory else 'enabled'}")
    print(f"  🔄 Number of runs: {args.num_runs}")
    print()

    print("\🖥️ System Configuration:")
    print(f"  System: {get_system()}")
    print(f"  CPU: {get_cpu_specs()}")
    print(f"  GPU: {get_gpu_specs()}")
    print()

    # Validate inputs
    if args.input_csv and not os.path.exists(args.input_csv):
        parser.error(f"CSV file not found: {args.input_csv}")

    if args.matrix_file and not os.path.exists(args.matrix_file):
        parser.error(f"Matrix file not found: {args.matrix_file}")

    if args.config_file:
        if not os.path.exists(args.config_file):
            parser.error(f"Config file not found: {args.config_file}")
    else:
        if not os.path.exists(args.config_dir):
            parser.error(f"Config directory not found: {args.config_dir}")
        config_files = [f for f in os.listdir(args.config_dir) if f.endswith('.json')]
        if not config_files:
            parser.error(f"No .json config files found in: {args.config_dir}")

    return args

def run_test(matrix_path, config_file, log_dir, use_cpu=False, pin_memory=True, k=5):
    """Run AMGX solver tests on a given matrix with specified configuration.
    
    Args:
        matrix_path (str): Path to the matrix file
        config_file (str): Path to the config file
        log_dir (str): Directory to store solver logs
        use_cpu (bool, optional): Whether to use CPU instead of GPU
        pin_memory (bool, optional): Whether to use pinned memory
        k (int, optional): Number of runs for averaging
    
    Returns:
        tuple: Contains (matrix_name, num_rows, elapsed_time, amgx_time, iterations, 
               solver_status, config_name) if successful, None if failed
    """
    matrix_name = os.path.basename(matrix_path).replace(".mtx", "")
    config_name = os.path.basename(config_file).replace(".json", "")
    log_file = os.path.join(log_dir, f"{matrix_name}_{config_name}.log")

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

        # **First run (always executed)**
        start_time = time.time()
        solver.initialize_matrix(row_ptr, col_indices, values)
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
                solver.initialize_matrix(row_ptr, col_indices, values)
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
        del solver

        solver_status = solver_statuses[-1]
        print(f"[RESULT] {matrix_name} ({config_name}): Num. Rows={num_rows}, Solver Status={solver_status}, "
              f"Avg Residual={amgx_residual}, Avg Iterations={avg_iterations}, "
              f"Avg Elapsed Time={avg_elapsed_time:.6f} s, Avg AMGX Time={avg_amgx_time:.6f} s")

        print("---------------------------------------------")
        return matrix_name, num_rows, avg_elapsed_time, avg_amgx_time, avg_iterations, solver_status, config_name, use_cpu, pin_memory, get_cpu_specs(), get_gpu_specs(), get_system()

    except Exception as e:
        print(f"[ERROR] Failed to solve {matrix_name} ({config_name}): {e}")
        return None

def main():
    """Main entry point for the AMGX solver testing pipeline."""
    args = parse_arguments()

    # Get list of matrices and configs
    matrix_files = get_matrix_files(args)
    if not matrix_files:
        print("[ERROR] No Matrix Market (.mtx) files found.")
        return

    # Get list of config files
    config_files = []
    if args.config_file:
        config_files = [args.config_file]
    else:
        config_files = sorted([os.path.join(args.config_dir, f) 
                             for f in os.listdir(args.config_dir) 
                             if f.endswith(".json")])

    print(f"[INFO] Testing {len(matrix_files)} matrices with {len(config_files)} configurations")

    # Run tests
    results = []
    for idx, matrix_path in enumerate(matrix_files, 1):
        matrix_name = os.path.basename(matrix_path).replace('.mtx', '')
        print(f"\n🔄 Processing matrix [{idx}/{len(matrix_files)}]: {matrix_name}")
        for config_file in config_files:
            result = run_test(matrix_path, config_file, args.log_dir,
                            use_cpu=args.use_cpu, 
                            pin_memory=not args.no_pin_memory, 
                            k=args.num_runs)
            if result:
                results.append(result)

    # Save results and generate plots
    if results:
        output_csv_path = os.path.join(args.output_dir, args.output_csv)
        save_results_to_csv(results, output_csv_path)
        # plot_results(output_csv_path, args.output_dir, "amgx")
    else:
        print("[WARNING] No results to plot or save.")

if __name__ == "__main__":
    main()
