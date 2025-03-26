#!/bin/bash
# 🛡️ Enable strict error handling
set -euo pipefail

# Function to log messages with timestamps
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Trap all uncaught errors
trap 'log_message "❌ An unexpected error occurred. Exiting..."; exit 1' ERR


# Function to run a test and check its exit status
run_test() {
    local test_name="$1"
    shift  # Remove first argument, leaving remaining args for the command
    
    log_message "🚀 Starting: $test_name"
    "$@"
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_message "✅ Completed: $test_name"
    else
        log_message "❌ Failed: $test_name (Exit code: $exit_code)"
        return $exit_code
    fi
}

# Set up environment
setup_environment() {
    log_message "Setting up environment variables..."

    CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}  # Default to 8 for local testing
    export OMP_NUM_THREADS=$CPUS
    export OPENBLAS_NUM_THREADS=$CPUS
    export MKL_NUM_THREADS=$CPUS
    export NUMEXPR_NUM_THREADS=$CPUS
    log_message "Threads: $CPUS"

    # Output directory
    export SHERLOCK_OUTPUT=sherlock_output
    export JOB_ID=${SLURM_JOB_ID:-local_run}
    mkdir -p "$SHERLOCK_OUTPUT/$JOB_ID"
    
    # Save system information
    log_message "Saving system information..."
    scontrol show node $SLURMD_NODENAME > $SHERLOCK_OUTPUT/$SLURM_JOB_ID/node_info_$SLURM_JOB_ID.txt
    lscpu > $SHERLOCK_OUTPUT/$SLURM_JOB_ID/cpu_info_$SLURM_JOB_ID.txt
    nvidia-smi > $SHERLOCK_OUTPUT/$SLURM_JOB_ID/gpu_info_$SLURM_JOB_ID.txt
    
    # Load modules
    log_message "Loading required modules (cuda, cmake, python)..."
    ml cuda/12 cmake/3.24 python/3.12
    
    log_message "Python version: $(python3 --version)"
    log_message "CUDA version:"
    nvcc --version | grep release


    # Move to repo directory
    cd "$GROUP_HOME/garaujor/BIODS253-Final-Project"

    # Build AMGX Solver
    ./build.sh

    # Activate virtual environment
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        log_message "❌ Virtual environment not found. Exiting..."
        exit 1
    fi
}

# Common arguments for all tests
COMMON_ARGS="--input_csv matrix_tests/matrices.csv --output_dir matrix_tests/output"

# Main execution
main() {
    setup_environment
    
    # Test 1: GPU AMGX solver
    run_test "GPU AMGX solver" python3 main.py \
        $COMMON_ARGS \
        --output_csv amgx_results.csv \
        --config_dir matrix_tests/configs
    
    # Test 2: CPU AMGX solver
    run_test "CPU AMGX solver" python3 main.py \
        $COMMON_ARGS \
        --use_cpu \
        --output_csv amgx_results_cpu.csv \
        --config_file matrix_tests/configs/minjie.json
    
    # Test 3: GPU AMGX solver without pinned memory
    run_test "GPU AMGX solver (no pinned memory)" python3 main.py \
        $COMMON_ARGS \
        --pin_memory 0 \
        --output_csv amgx_results_no_pin.csv \
        --config_file matrix_tests/configs/minjie.json
    
    # Test 4: SciPy CG solver
    run_test "SciPy CG solver" python3 main_scipy.py \
        $COMMON_ARGS \
        --output_csv scipy_results.csv
    
    log_message "🎉 All tests completed successfully!"
}

# Run main function and capture any errors
if ! main; then
    log_message "❌ Job failed with errors"
    exit 1
fi