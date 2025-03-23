#!/bin/bash
# Exit immediately if any command returns a non-zero (error) status.
set -e

SECONDS=0
# Save the starting directory
PARENT_DIR=$(pwd)  

# Detect OS
OS_TYPE="$(uname -s)"
case "$OS_TYPE" in
    Linux*)     PLATFORM="Linux";;
    Darwin*)    echo "macOS is not supported. Exiting."; exit 1;;
    MINGW*|MSYS*|CYGWIN*) PLATFORM="Windows";;
    *)          echo "Unknown OS: $OS_TYPE. Exiting."; exit 1;;
esac

echo "Detected OS: $PLATFORM"

# Load modules if on cluster (Linux HPC-like environment)
echo "NOTE: Make sure the following modules are loaded appropriately for your system and added to your path:"
if [[ "$PLATFORM" == "Linux" ]]; then
  echo "module load cuda/12.6 cmake/3.24 python/3.12"
  module load cuda/12.6 cmake/3.24 python/3.12  # <- change as needed
else
  echo "cuda/12.x.x cmake/3.24 python/3.x.x"
fi

# Choose correct Python interpreter
if command -v python3 &> /dev/null; then
  PYTHON=python3
elif command -v python &> /dev/null && [[ "$($PYTHON --version 2>&1)" == "Python 3"* ]]; then
  PYTHON=python
else
  echo "❌ No suitable Python 3 interpreter found. Exiting."
  exit 1
fi

echo "📦 Using Python interpreter: $($PYTHON --version)"


# Check if AmgX is outdated
AMGX_DIR="extern/amgx"
cd "$AMGX_DIR"
echo "Checking for updates to AmgX..."
git fetch origin
LOCAL_HASH=$(git rev-parse HEAD)
git remote set-head origin --auto
REMOTE_HASH=$(git rev-parse origin/HEAD)
cd "$PARENT_DIR" > /dev/null # Go back to the original directory 

AMGX_LIB_LINUX="$AMGX_DIR/build/libamgxsh.so"
AMGX_LIB_WINDOWS="$AMGX_DIR/build/Release/amgxsh.dll"

if [[ "$LOCAL_HASH" != "$REMOTE_HASH" || ( ! -f "$AMGX_LIB_LINUX" && ! -f "$AMGX_LIB_WINDOWS" ) ]]; then
  echo "🔄 AmgX appears to be outdated or not yet built."

  cd "$AMGX_DIR"

  # Ensure submodules are pulled (in case user forgot --recursive)
  echo "Ensuring AmgX submodules are initialized..."
  git submodule update --init --recursive

  # Get default branch (main or master, etc.)
  DEFAULT_BRANCH=$(git remote show origin | awk '/HEAD branch/ {print $NF}')
  git checkout "$DEFAULT_BRANCH"
  git pull origin "$DEFAULT_BRANCH"

  rm -rf build # VERY IMPORTANT!
  mkdir -p build && cd build

  # Configure build system
  if [[ "$PLATFORM" == "Windows" ]]; then
    echo "🔧 Configuring AmgX with Visual Studio generator..."
    cmake .. -DCMAKE_BUILD_TYPE=Release
    echo "🏗️ Building AmgX with cmake --build"
    cmake --build . --config Release --parallel
  else
    echo "🔧 Configuring AmgX with Unix Makefiles..."
    cmake .. -DCMAKE_BUILD_TYPE=Release
    echo "🏗️ Building AmgX with make"
    make -j$(nproc) all
  fi

  cd "$PARENT_DIR"  # Go back to the original directory
else
  echo "✅ AmgX is up to date. Skipping rebuild."
fi

# Set up Python environment
ENV_DIR="venv"
if [ ! -d "$ENV_DIR" ]; then
  echo "Creating Python virtual environment in $ENV_DIR..."
  $PYTHON -m venv "$ENV_DIR"
fi

# Determine activate script path based on OS
if [[ "$PLATFORM" == "Windows" ]]; then
  ACTIVATE_FILE="$ENV_DIR/Scripts/activate"
else
  ACTIVATE_FILE="$ENV_DIR/bin/activate"
fi

# Add PYTHONPATH line if not already in activate script
PYTHONPATH_LINE="export PYTHONPATH=\"$PARENT_DIR/build:\$PYTHONPATH\""
if ! grep -Fxq "$PYTHONPATH_LINE" "$ACTIVATE_FILE"; then
  echo "$PYTHONPATH_LINE" >> "$ACTIVATE_FILE"
  echo "✅ PYTHONPATH updated in venv activation script."
fi

# Activate the virtual environment
if [ ! -f "$ACTIVATE_FILE" ]; then
  echo "❌ Cannot find activation script at $ACTIVATE_FILE"
  exit 1
fi
source "$ACTIVATE_FILE"
echo "🐍 Activated Python environment: $(which python)"

$PYTHON -m pip install --upgrade pip

if [ ! -f requirements.txt ]; then
  echo "⚠️ requirements.txt not found. Check that you have the latest version of the repo."
  exit 1
else
  $PYTHON -m pip install -r requirements.txt
fi

# Build your main app
rm -rf build # VERY IMPORTANT!
mkdir -p build && cd build
cmake ..
make -j$(nproc)

echo "✅ Build complete."
echo "🕒 Build finished in ${SECONDS}s"
