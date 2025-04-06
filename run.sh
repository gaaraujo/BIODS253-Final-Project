#!/bin/bash
# 🛡️ Enable strict error handling
set -euo pipefail

PARENT_DIR=$(pwd)  

# Detect OS
OS_TYPE="$(uname -s)"
case "$OS_TYPE" in
    Linux*)     PLATFORM="Linux";;
    Darwin*)    echo "macOS is not supported. Exiting."; exit 1;;
    MINGW*|MSYS*|CYGWIN*) PLATFORM="Windows";;
    *)          echo "Unknown OS: $OS_TYPE. Exiting."; exit 1;;
esac

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

    CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)} 
    export OMP_NUM_THREADS=$CPUS
    export OPENBLAS_NUM_THREADS=$CPUS
    export MKL_NUM_THREADS=$CPUS
    export NUMEXPR_NUM_THREADS=$CPUS
    log_message "Threads: $CPUS"

    # Output directory
    export SHERLOCK_OUTPUT=sherlock_output
    JOB_ID=${SLURM_JOB_ID:-local_run}
    export JOB_ID
    
    # Load modules if `ml` is available (required on Sherlock cluster)
   if [[ "$PLATFORM" == "Linux" ]] && command -v ml &>/dev/null; then
        log_message "Loading required modules (cuda, cmake, python)..."
        ml cuda/12 cmake/3.24 python/3.12
    else
        log_message "Please make sure cuda, cmake, and python are available."
    fi
    
    # Choose correct Python interpreter
    PYTHON=""
    if command -v python3 &> /dev/null && [[ "$(python3 --version 2>&1)" == "Python 3."* ]]; then
        PYTHON=python3
    elif command -v python &> /dev/null && [[ "$(python --version 2>&1)" == "Python 3."* ]]; then
        PYTHON=python
    else
        echo "❌ No suitable Python 3 interpreter found."
        echo "🔧 Please manually set the PYTHON variable near the top of this script to your Python 3 path."
        echo "💡 For example:"
        echo '    PYTHON="/c/Users/yourname/AppData/Local/anaconda3/python.exe"'
        echo "📝 Then comment out the automatic interpreter detection block."
        exit 1
    fi
    export PYTHON
    log_message "Python version: $($PYTHON --version)"
    log_message "CUDA version:"
    command -v nvcc &>/dev/null && nvcc --version | grep release || echo "nvcc not available"


    # Move to repo directory
    cd "$PARENT_DIR"

    # Build AMGX Solver
    log_message "Building AMGX Solver..."
    ./build.sh

    # Activate virtual environment
    log_message "Activating virtual environment..."

    if [ "$PLATFORM" == "Windows" ]; then
        ACTIVATE_PATH="venv/Scripts/activate"
    else
        ACTIVATE_PATH="venv/bin/activate"
    fi

    if [ -f "$ACTIVATE_PATH" ]; then
        source "$ACTIVATE_PATH"
    else
        log_message "❌ Virtual environment not found at $ACTIVATE_PATH. Exiting..."
        exit 1
    fi

}

# Main execution
main() {
    setup_environment
    
    # Common arguments for all tests
    COMMON_ARGS="--input_csv matrix_tests/matrices.csv --output_dir matrix_tests/output_$JOB_ID"

    # Test 1: GPU AMGX solver
    run_test "GPU AMGX solver" $PYTHON main.py \
        $COMMON_ARGS \
        --output_csv amgx_results.csv \
        --config_dir matrix_tests/configs
    
    # Test 2: GPU AMGX solver without pinned memory
    run_test "GPU AMGX solver (no pinned memory)" $PYTHON main.py \
        $COMMON_ARGS \
        --no_pin_memory \
        --output_csv amgx_results_no_pin.csv \
        --config_file matrix_tests/configs/minjie.json
    
    log_message "🎉 All GPU tests completed successfully!"
}

# Run main function and capture any errors
if ! main; then
    log_message "❌ Job failed with errors"
    exit 1
fi