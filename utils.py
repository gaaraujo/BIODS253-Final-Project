import os
import numpy as np
import scipy.io
import scipy.sparse as sp
import urllib.request
import tarfile
import shutil
import pandas as pd
import matplotlib.pyplot as plt

# Base URL for downloading SuiteSparse matrices
BASE_URL = "https://suitesparse-collection-website.herokuapp.com/MM"

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
    
    return A.tocsr()  # Convert to CSR format

def download_and_extract(group, matrix_name, matrices_dir):
    """Download and extract a SuiteSparse matrix in .mtx format if it's missing.

    Downloads matrix from SuiteSparse collection, extracts it, and saves in the
    specified matrix directory. Cleans up temporary files after extraction.

    Args:
        group (str): Matrix group name in SuiteSparse collection
        matrix_name (str): Name of the matrix to download
        matrices_dir (str): Directory to save the matrix
    """
    file_url = f"{BASE_URL}/{group}/{matrix_name}.tar.gz"
    archive_path = os.path.join(matrices_dir, f"{matrix_name}.tar.gz")
    extract_path = os.path.join(matrices_dir, matrix_name)
    final_mtx_path = os.path.join(matrices_dir, f"{matrix_name}.mtx")

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

def get_matrix_files(args):
    """Get list of matrix files based on input arguments.
    
    Args:
        args: Parsed command line arguments containing:
            - input_csv: Path to CSV file with matrix metadata
            - matrix_file: Path to single matrix file
            - matrices_dir: Directory containing matrices
    
    Returns:
        list: Paths to matrix files to process
    """
    matrix_files = []
    if args.input_csv:
        # Download and collect matrices from CSV
        print(f"📥 Downloading matrices specified in {args.input_csv} to {args.matrices_dir}")
        df = pd.read_csv(args.input_csv)
        for _, row in df.iterrows():
            matrix_name = row["Name"]
            group = row["Group"]
            if not pd.isna(matrix_name) and not pd.isna(group):
                download_and_extract(group, matrix_name, args.matrices_dir)
                mtx_path = os.path.join(args.matrices_dir, f"{matrix_name}.mtx")
                if os.path.exists(mtx_path):
                    matrix_files.append(mtx_path)
        matrix_files.sort()  # Keep files sorted
    elif args.matrix_file:
        matrix_files = [args.matrix_file]
    else:
        print(f"ℹ️  Looking for .mtx files in: {args.matrices_dir}")
        matrix_files = sorted([os.path.join(args.matrices_dir, f) 
                             for f in os.listdir(args.matrices_dir) 
                             if f.endswith(".mtx")])
    return matrix_files

def save_results_to_csv(results, output_csv):
    """Save test results to CSV file.
    
    Args:
        results (list): List of tuples containing:
            (matrix_name, num_rows, elapsed_time, solve_time, iterations, status, config_name)
        output_csv (str): Path to save the results CSV
    """
    data = {
        "Matrix": [],
        "Config": [],
        "NumRows": [],
        "SolveTime": [],
        "Iterations": [],
        "Success": []
    }

    for result in results:
        matrix_name, num_rows, _, solve_time, iterations, status, config_name = result
        
        data["Matrix"].append(matrix_name)
        data["Config"].append(config_name)
        data["NumRows"].append(num_rows)
        data["SolveTime"].append(solve_time)
        data["Iterations"].append(iterations)
        data["Success"].append(1 if status == 0 else 0)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Test results saved to {output_csv}")


def plot_results(csv_file, plot_dir, prefix):
    """Plot performance metrics against matrix sizes from CSV results.

    Args:
        csv_file (str): Path to the CSV file containing results
        plot_dir (str): Directory to save the plot files
        prefix (str): Unique prefix for plots
    """
    # Read results from CSV
    df = pd.read_csv(csv_file)

    # Dictionary to store results for plotting
    data = {}

    # Group by config and process each group
    for config_name, group in df.groupby('Config'):
            data[config_name] = {
            "sizes": group[group['Success'] == 1]['NumRows'].values,
            "amgx_times": group[group['Success'] == 1]['SolveTime'].values,
            "iterations": group[group['Success'] == 1]['Iterations'].values,
            "failed_sizes": group[group['Success'] == 0]['NumRows'].values,
            "failed_amgx_times": group[group['Success'] == 0]['SolveTime'].values,
            "failed_iterations": group[group['Success'] == 0]['Iterations'].values
        }

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
        if len(sorted_failed_sizes) > 0:
            plt.scatter(sorted_failed_sizes, sorted_failed_amgx_times, marker='x', 
                       color=line.get_color(), label=f"{config_name} (Failed)")

    plt.xlabel("Matrix Size (Number of Rows)")
    plt.ylabel("Solve Time (s)")
    plt.title("Solve Time vs Matrix Size")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"{prefix}_solve_time.png"))
    plt.close()

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
        if len(sorted_failed_sizes) > 0:
            plt.scatter(sorted_failed_sizes, sorted_failed_iterations, marker='x', 
                       color=line.get_color(), label=f"{config_name} (Failed)")

    plt.xlabel("Matrix Size (Number of Rows)")
    plt.ylabel("Number of Iterations")
    plt.title("Iterations vs Matrix Size")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"{prefix}_iterations_vs_matrix_size.png"))
    plt.close()

    print(f"✅ Plots saved in {plot_dir}")

def import_pyAMGXSolver():
    """Import pyAMGXSolver module after setting up necessary paths.
    
    Returns:
        module: The imported pyAMGXSolver module
    
    Raises:
        RuntimeError: If build directory is not found
    """
    import os
    import sys
    
    # Add the build directory to Python path
    build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "build"))
    if os.path.exists(build_dir):
        # Add to Python path
        if build_dir not in sys.path:
            sys.path.insert(0, build_dir)  # Insert at beginning to ensure it's checked first
        
        # Add to Windows DLL search path
        if sys.platform == 'win32':
            try:
                os.add_dll_directory(build_dir)
                
                # Add CUDA path if available
                cuda_path = os.environ.get('CUDA_PATH')
                if cuda_path:
                    cuda_bin = os.path.join(cuda_path, 'bin')
                    if os.path.exists(cuda_bin):
                        os.add_dll_directory(cuda_bin)
            except Exception as e:
                print(f"Warning: Could not add DLL directory: {e}")
    else:
        raise RuntimeError(f"Build directory not found at {build_dir}. Please build the project first.")

    import pyAMGXSolver
    return pyAMGXSolver