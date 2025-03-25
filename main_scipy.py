import os
import time
import numpy as np
import scipy.sparse.linalg as spla
from utils import (
    load_matrix, get_matrix_files,
    save_results_to_csv, plot_results
)
import argparse

def parse_arguments():
    """Parse command-line arguments for directory paths and solver configuration.
    
    Matrix input modes:
    1. With --input_csv: Downloads matrices specified in CSV to matrices_dir
    2. Without --input_csv: Uses existing .mtx files from matrices_dir
    3. Single matrix: Uses specified matrix file
    """
    parser = argparse.ArgumentParser(description="Run SciPy CG solver tests on matrices.")
    
    # Matrix inputs
    matrix_group = parser.add_mutually_exclusive_group()
    matrix_group.add_argument("--input_csv", type=str,
                            help="CSV file containing matrix metadata for downloading matrices")
    matrix_group.add_argument("--matrix_file", type=str,
                            help="Path to single matrix file to test")

    # Directories
    parser.add_argument("--matrices_dir", type=str, default="matrix_tests/matrices",
                        help="Directory containing/storing matrix files (.mtx)")
    parser.add_argument("--output_dir", type=str, default="matrix_tests/output",
                        help="Directory to save results and plots")
    parser.add_argument("--output_csv", type=str, default="scipy_results.csv",
                        help="Name of the CSV file to save in output_dir")

    # Solver configuration
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of runs for computing average performance")

    args = parser.parse_args()

    # Validate output_csv is just a filename, not a path
    if os.path.sep in args.output_csv:
        parser.error("--output_csv should be a filename, not a path. It will be saved in output_dir.")

    # Create necessary directories
    os.makedirs(args.matrices_dir, exist_ok=True)
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

    print("\nOutput Locations:")
    print(f"  📂 Output directory: {args.output_dir}")
    print(f"  📊 Results CSV: {os.path.join(args.output_dir, args.output_csv)}")
    
    print("\nSolver Configuration:")
    print(f"  🔄 Number of runs: {args.num_runs}")
    print()

    # Validate inputs
    if args.input_csv and not os.path.exists(args.input_csv):
        parser.error(f"CSV file not found: {args.input_csv}")

    if args.matrix_file and not os.path.exists(args.matrix_file):
        parser.error(f"Matrix file not found: {args.matrix_file}")

    if not args.matrix_file and not os.path.exists(args.matrices_dir):
        parser.error(f"Matrices directory not found: {args.matrices_dir}")

    return args

def diagonal_jacobi_preconditioner(A):
    """Create a diagonal Jacobi preconditioner for the given matrix."""
    M_inv = 1.0 / A.diagonal()
    return spla.LinearOperator(A.shape, matvec=lambda x: M_inv * x)

def cg_solve(A, b, M):
    """Run CG solver with iteration counting and timing.
    
    Args:
        A: sparse matrix
        b: right-hand side vector
        M: preconditioner
    
    Returns:
        tuple: Contains (solve_time, iterations, residual, status, solution)
    """
    num_iterations = [0]
    def callback(xk):
        num_iterations[0] += 1

    start_time = time.time()
    x, status = spla.cg(A, b, M=M, callback=callback, atol=1e-6, rtol=0., maxiter=None)
    solve_time = time.time() - start_time

    residual = np.linalg.norm(A @ x - b)
    
    return solve_time, num_iterations[0], residual, status, x

def run_test(matrix_path, k=5):
    """Run SciPy CG solver test on a given matrix."""
    matrix_name = os.path.basename(matrix_path).replace(".mtx", "")
    print("---------------------------------------------")
    try:
        A = load_matrix(matrix_path)
        num_rows = A.shape[0]
        print(f"[TEST] {matrix_name}: Num. Rows={num_rows}")

        # Generate right-hand side vector
        rng = np.random.default_rng(42)
        b = rng.random(num_rows, dtype=np.float64)

        # Create preconditioner
        M = diagonal_jacobi_preconditioner(A)

        # First run to check convergence
        solve_time, iterations, residual, status, x = cg_solve(A, b, M)

        # If solver didn't converge, return without additional runs
        if status != 0:
            print(f"[RESULT] {matrix_name}: Num. Rows={num_rows}, Status={status}, "
                  f"Iterations={iterations}, "
                  f"Residual={residual:.6e}, Solve Time={solve_time:.6f} s")
            print("---------------------------------------------")
            return matrix_name, num_rows, solve_time, solve_time, iterations, status, "scipy_cg_diagjacobi"

        # If we got here, solver converged, do remaining k-1 runs
        solve_times = []
        residuals = []
        iterations_list = []

        for _ in range(k - 1):
            solve_time, iterations, residual, status, x = cg_solve(A, b, M)
            solve_times.append(solve_time)
            residuals.append(residual)
            iterations_list.append(iterations)

        # Compute averages (excluding first run)
        avg_solve_time = np.mean(solve_times)
        avg_residual = np.mean(residuals)
        avg_iterations = np.mean(iterations_list)

        print(f"[RESULT] {matrix_name}: Num. Rows={num_rows}, Status={status}, "
              f"Avg Iterations={avg_iterations:.1f}, "
              f"Avg Residual={avg_residual:.6e}, Avg Solve Time={avg_solve_time:.6f} s")
        print("---------------------------------------------")

        return matrix_name, num_rows, avg_solve_time, avg_solve_time, avg_iterations, status, "scipy_cg_diagjacobi"

    except Exception as e:
        print(f"[ERROR] Failed to solve {matrix_name}: {e}")
        return None

def main():
    """Main entry point for the SciPy solver testing pipeline."""
    args = parse_arguments()

    # Get list of matrices
    matrix_files = get_matrix_files(args)
    if not matrix_files:
        print("[ERROR] No Matrix Market (.mtx) files found.")
        return

    print(f"[INFO] Testing {len(matrix_files)} matrices with SciPy CG solver")

    # Run tests
    results = []
    for idx, matrix_path in enumerate(matrix_files, 1):
        matrix_name = os.path.basename(matrix_path).replace('.mtx', '')
        print(f"\n🔄 Processing matrix [{idx}/{len(matrix_files)}]: {matrix_name}")
        result = run_test(matrix_path, k=args.num_runs)
        if result:
            results.append(result)

    # Save results and generate plots
    if results:
        output_csv_path = os.path.join(args.output_dir, args.output_csv)
        save_results_to_csv(results, output_csv_path)
        plot_results(output_csv_path, args.output_dir, "scipy_cg")
    else:
        print("[WARNING] No results to plot or save.")

if __name__ == "__main__":
    main()
