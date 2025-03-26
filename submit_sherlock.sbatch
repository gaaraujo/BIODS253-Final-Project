#!/bin/bash
#SBATCH --job-name=AMGX_Profile
#SBATCH -p gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=sherlock_output/%j/AMGX_Profile.%j.out
#SBATCH --error=sherlock_output/%j/AMGX_Profile.%j.err

# GPU and CPU configuration
#SBATCH --constraint 'GPU_SKU:H100_SXM5&CPU_SKU:8462Y+'

# Some examples of available devices:
# - Hopper H100_SXM5 vs Intel Xeon Platinum 8462Y+
#   #SBATCH --constraint 'GPU_SKU:H100_SXM5&CPU_SKU:8462Y+'
# - Lovelace L40S vs Intel Xeon 6426Y
#   #SBATCH --constraint 'GPU_SKU:L40S&CPU_SKU:6426Y'
# - Volta V100_SXM2 vs Intel Xeon Gold 5118
#   #SBATCH --constraint 'GPU_SKU:V100_SXM2&CPU_SKU:5118'
# - Pascal P100_PCIE vs Intel Xeon E5-2640v
#   #SBATCH --constraint 'GPU_SKU:P100_PCIE&CPU_SKU:E5-2640v'

#SBATCH --mail-user=garaujor@stanford.edu 
#SBATCH --mail-type=BEGIN,END,FAIL

# Run the test
./run.sh

# Test 3: GPU AMGX solver without pinned memory
run_test "GPU AMGX solver (no pinned memory)" python3 main.py \
    $COMMON_ARGS \
    --no_pin_memory \
    --output_csv amgx_results_no_pin.csv \
    --config_file matrix_tests/configs/minjie.json