import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pyAMGXSolver
import os
import json

# Default tolerance and config file
DEFAULT_TOLERANCE = 1e-6
CONFIG_FILE = "config.json"

def create_default_config_file(tolerance=DEFAULT_TOLERANCE):
    """
    Creates a default AMGX configuration file dynamically.
    """
    config_data = {
        "config_version": 2,
        "solver": {
            "preconditioner": {
                "scope": "precond",
                #"solver": "MULTICOLOR_DILU"
                "solver": "BLOCK_JACOBI"
            },
            "solver": "PCG",
            "print_solve_stats": 1,
            "obtain_timings": 1,
            "max_iters": 20,
            "monitor_residual": 1, 
            "store_res_history": 1,
            "scope": "main",
            "tolerance": tolerance,
            "norm": "L2"
        }
    }

    with open(CONFIG_FILE, "w") as f:
        json.dump(config_data, f, indent=4)
    
    print(f"[INFO] Created config file: {CONFIG_FILE} with tolerance {tolerance}")

def create_test_matrix(size):
    """
    Creates a symmetric positive definite (SPD) tridiagonal matrix in CSR format.
    Returns row_ptr, col_indices, values (CSR format), and a right-hand side vector b.
    """
    diagonals = [2.0 * np.ones(size), -1.0 * np.ones(size - 1), -1.0 * np.ones(size - 1)]
    offsets = [0, -1, 1]
    A = sp.diags(diagonals, offsets, shape=(size, size), format="csr")

    # Convert to CSR format manually
    row_ptr = A.indptr.astype(np.int32)
    col_indices = A.indices.astype(np.int32)
    values = A.data.astype(np.float64)
    
    # Define right-hand side vector
    b = np.ones(size, dtype=np.float64)

    return row_ptr, col_indices, values, b, A

def run_test(size, use_cpu=False, pin_memory=True, log_file=None):
    """
    Runs a test case for a given matrix size and configuration.
    Solves Ax=b using pyAMGXSolver and compares with SciPy's solver.
    """
    print(f"\n[INFO] Running test with matrix size: {size}x{size}, use_cpu={use_cpu}, pin_memory={pin_memory}, log_file={log_file}")

    # Generate test matrix
    row_ptr, col_indices, values, b, A = create_test_matrix(size)

    # Initialize AMGX solver
    solver = pyAMGXSolver.AMGXSolver(CONFIG_FILE, use_cpu=use_cpu, gpu_ids=[0], pin_memory=pin_memory, log_file=log_file)
    solver.initialize_matrix(row_ptr, col_indices, values)

    # Solve using AMGX
    x_amgx, status, iterations, residual = solver.solve(b)

    del solver # to avoid memory leaks

    # Solve using SciPy for reference
    x_scipy = spla.spsolve(A, b)

    # Compute residual norm ||Ax - b||
    residual_norm_amgx = np.linalg.norm(A @ x_amgx - b)
    residual_norm_scipy = np.linalg.norm(A @ x_scipy - b)

    # Compare AMGX and SciPy results
    max_diff = np.max(np.abs(x_amgx - x_scipy))
    
    print(f"AMGX Residual Norm: {residual_norm_amgx:.6e}")
    print(f"SciPy Residual Norm: {residual_norm_scipy:.6e}")
    print(f"Max difference between AMGX and SciPy solutions: {max_diff:.6e}")

    # Check results
    if residual_norm_amgx < DEFAULT_TOLERANCE and max_diff < 1e-6:
        print(f"[PASS] Test passed for size {size}")
        return True
    else:
        print(f"[FAIL] Test failed for size {size}")
        return False

def test_expected_exceptions():
    """
    Runs tests to check that expected exceptions are raised for invalid inputs.
    """
    print("\n---------------------------------")
    print("[INFO] Running expected exception tests...")

    num_rows = 5
    row_ptr, col_indices, values, b, _ = create_test_matrix(num_rows)

    failed_exceptions = 0
    passed_exceptions = 0

    # 1. Test: Null/empty config file
    print("---------------------------------")
    print("[TEST] Null/empty config file.")
    try:
        solver = pyAMGXSolver.AMGXSolver("", gpu_ids=[0])
        print("[FAIL] Expected exception for empty config file not thrown.")
        failed_exceptions += 1
    except Exception as e:
        solver = None # to avoid memory leaks
        print(f"[PASS] Caught expected exception: {e}")
        passed_exceptions += 1

    # 2. Test: Null gpu_ids
    print("---------------------------------")
    print("[TEST] Null/empty array of GPU IDs.")
    try:
        solver = pyAMGXSolver.AMGXSolver(CONFIG_FILE)
        print("[FAIL] Expected exception for empty array of GPU IDs not thrown.")
        failed_exceptions += 1
    except Exception as e:
        solver = None # to avoid memory leaks
        print(f"[PASS] Caught expected exception: {e}")
        passed_exceptions += 1

    # 3. Test: Null row_ptr
    print("---------------------------------")
    print("[TEST] Null row_ptr.")
    try:
        solver = pyAMGXSolver.AMGXSolver(CONFIG_FILE, gpu_ids=[0])
        solver.initialize_matrix(None, col_indices, values)
        print("[FAIL] Expected exception for null row_ptr not thrown.")
        failed_exceptions += 1
    except Exception as e:
        solver = None # to avoid memory leaks
        print(f"[PASS] Caught expected exception: {e}")
        passed_exceptions += 1

    # 4. Test: Null col_indices
    print("---------------------------------")
    print("[TEST] Null col_indices.")
    try:
        solver = pyAMGXSolver.AMGXSolver(CONFIG_FILE, gpu_ids=[0])
        invalid_col_indices = np.array([0, 1, 5, 2, 10], dtype=np.int32)  # Index out of bounds
        solver.initialize_matrix(row_ptr, None, values)
        print("[FAIL] Expected exception for null col_indices not thrown.")
        failed_exceptions += 1
    except Exception as e:
        solver = None # to avoid memory leaks
        print(f"[PASS] Caught expected exception: {e}")
        passed_exceptions += 1

    # 5. Test: Null values array
    print("---------------------------------")
    print("[TEST] Null values array.")
    try:
        solver = pyAMGXSolver.AMGXSolver(CONFIG_FILE, gpu_ids=[0])
        solver.initialize_matrix(row_ptr, col_indices, None)
        print("[FAIL] Expected exception for null values not thrown.")
        failed_exceptions += 1
    except Exception as e:
        solver = None # to avoid memory leaks
        print(f"[PASS] Caught expected exception: {e}")
        passed_exceptions += 1

    # 6. Test: replace_coefficients with mismatching num_rows
    print("---------------------------------")
    print("[TEST] replace_coefficients with mismatching num_rows.")
    try:
        solver = pyAMGXSolver.AMGXSolver(CONFIG_FILE, gpu_ids=[0])
        solver.initialize_matrix(row_ptr, col_indices, values)
        solver.replace_coefficients(values[:-1])
        print("[FAIL] Expected exception for mistmatching size of values not thrown.")
        failed_exceptions += 1
    except Exception as e:
        solver = None # to avoid memory leaks
        print(f"[PASS] Caught expected exception: {e}")
        passed_exceptions += 1

    # 7. Test: replace_coefficients before initialization
    print("---------------------------------")
    print("[TEST] replace_coefficients before initialization.")
    try:
        solver = pyAMGXSolver.AMGXSolver(CONFIG_FILE, gpu_ids=[0])
        solver.replace_coefficients(values)
        print("[FAIL] Expected exception for uninitialized matrix values not thrown.")
        failed_exceptions += 1
    except Exception as e:
        solver = None # to avoid memory leaks
        print(f"[PASS] Caught expected exception: {e}")
        passed_exceptions += 1

    # 8. Test: replace_coefficients with null values
    print("---------------------------------")
    print("[TEST] replace_coefficients with null values.")
    try:
        solver = pyAMGXSolver.AMGXSolver(CONFIG_FILE, gpu_ids=[0])
        solver.initialize_matrix(row_ptr, col_indices, values)
        solver.replace_coefficients(None)
        print("[FAIL] Expected exception for null values not thrown.")
        failed_exceptions += 1
    except Exception as e:
        solver = None # to avoid memory leaks
        print(f"[PASS] Caught expected exception: {e}")
        passed_exceptions += 1

    # 9. Test: Solve before initialization
    print("---------------------------------")
    print("[TEST] Solve before initialization.")
    try:
        solver = pyAMGXSolver.AMGXSolver(CONFIG_FILE, gpu_ids=[0])
        solver.solve(b)
        print("[FAIL] Expected exception for solving before initializing matrix not thrown.")
        failed_exceptions += 1
    except Exception as e:
        solver = None # to avoid memory leaks
        print(f"[PASS] Caught expected exception: {e}")
        passed_exceptions += 1

    print("=====================================")
    print("Exception Tests Summary:")
    print(f"✅ Passed: {passed_exceptions}, ❌ Failed: {failed_exceptions}")
    print("=====================================")

def main():
    """
    Runs tests for different matrix sizes and configurations.
    """
    create_default_config_file()

    test_sizes = [5, 50, 200]  # Small, medium, large

    # Different configurations
    use_cpu_options = [False, True]
    pin_memory_options = [False, True]
    log_files = [None, "solver_log.txt"]

    passed_tests = 0
    failed_tests = 0

    for use_cpu in use_cpu_options:
        for pin_memory in pin_memory_options:
            for log_file in log_files:
                print("\n=====================================")
                print(f"[INFO] Running tests with use_cpu={use_cpu}, pin_memory={pin_memory}, log_file={log_file}")
                print("=====================================")

                for size in test_sizes:
                    if run_test(size, use_cpu, pin_memory, log_file):
                        passed_tests += 1
                    else:
                        failed_tests += 1

                # Remove log file if created
                if log_file and os.path.exists(log_file):
                    os.remove(log_file)
                    print(f"[INFO] Deleted log file: {log_file}")

    # Remove config file after tests
    if os.path.exists(CONFIG_FILE):
        os.remove(CONFIG_FILE)
        print(f"[INFO] Deleted config file: {CONFIG_FILE}")

    # Test for Exceptions
    create_default_config_file()
    test_expected_exceptions()

    # Print summary
    print("\n=====================================")
    print(f"✅ Solver Tests Passed: {passed_tests}")
    print(f"❌ Solver Tests Failed: {failed_tests}")
    print("=====================================")

    return 0 if failed_tests == 0 else 1

if __name__ == "__main__":
    exit(main())

