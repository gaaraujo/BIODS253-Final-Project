cmake_minimum_required(VERSION 3.17)
project(AMGXProject)

# Require C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Detect OS and set library file name
if(APPLE)
    message(FATAL_ERROR "AMGX is not supported on macOS. Compilation aborted.")
elseif(WIN32)
    set(AMGX_LIB_NAME "amgxsh.dll")
else()  # Default to Linux
    set(AMGX_LIB_NAME "libamgxsh.so")
endif()

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)

# Set AMGX paths: use custom path if specified, otherwise use the default
set(AMGX_CUSTOM_PATH "" CACHE PATH "Custom AMGX installation path")

if(AMGX_CUSTOM_PATH)
    set(AMGX_ROOT_DIR "${AMGX_CUSTOM_PATH}")
else()
    set(AMGX_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/amgx")
endif()

set(AMGX_INCLUDE_DIR "${AMGX_ROOT_DIR}/include")
set(AMGX_LIBRARY_DIR "${AMGX_ROOT_DIR}/lib")

# Ensure AMGX headers and library exist
if(NOT EXISTS "${AMGX_INCLUDE_DIR}/amgx_c.h")
    message(FATAL_ERROR "AMGX header not found: ${AMGX_INCLUDE_DIR}/amgx_c.h. Set AMGX_CUSTOM_PATH correctly.")
endif()
if(NOT EXISTS "${AMGX_LIBRARY_DIR}/${AMGX_LIB_NAME}")
    message(FATAL_ERROR "AMGX library not found at ${AMGX_LIBRARY_DIR}/${AMGX_LIB_NAME}")
endif()

# Include AMGX headers
include_directories(${AMGX_INCLUDE_DIR})

# Create library from AMGXSolver
add_library(solver SHARED ${CMAKE_CURRENT_SOURCE_DIR}/SRC/solver/AMGXSolver.cpp)
target_include_directories(solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/SRC/solver)
target_link_libraries(solver PRIVATE ${AMGX_LIBRARY_DIR}/libamgxsh.so)

# Create tests executable
add_executable(test_solver ${CMAKE_CURRENT_SOURCE_DIR}/SRC/tests/test_solver.cpp)
target_include_directories(test_solver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/SRC/solver)
target_link_libraries(test_solver PRIVATE solver)

# Create Python interface for AMGXSolver
pybind11_add_module(pyAMGXSolver ${CMAKE_CURRENT_SOURCE_DIR}/SRC/solver/pyAMGXSolver.cpp)
target_link_libraries(pyAMGXSolver PRIVATE solver)

# Print summary
message(STATUS "AMGX found at: ${AMGX_ROOT_DIR}")
message(STATUS "CUDA found at: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "Building AMGX solver with C++11 support.")
