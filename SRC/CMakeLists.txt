cmake_minimum_required(VERSION 3.17)
project(AMGXProject)

# Require C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find CUDA (modern method)
find_package(CUDAToolkit REQUIRED)

# Set AMGX paths
set(AMGX_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/amgx")
set(AMGX_INCLUDE_DIR "${AMGX_ROOT_DIR}/include")
set(AMGX_LIBRARY_DIR "${AMGX_ROOT_DIR}/lib")

# Ensure AMGX library exists
if(NOT EXISTS "${AMGX_LIBRARY_DIR}/libamgx.so")
    message(FATAL_ERROR "AMGX library not found at ${AMGX_LIBRARY_DIR}/libamgx.so")
endif()

# Define test data path
set(TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test_data")

# Create library from AMGXSolver
add_library(solver solver/AMGXSolver.cpp)

# Link AMGX to solver
target_link_libraries(solver PRIVATE ${AMGX_LIBRARY_DIR}/libamgx.so)

# Add solver include directories
target_include_directories(solver PRIVATE ${AMGX_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Add tests
add_executable(test_solver tests/test_solver.cpp)

# Link test_solver with solver
target_link_libraries(test_solver PRIVATE solver)

# Define test data path for test_solver
target_compile_definitions(test_solver PRIVATE TEST_DATA_DIR="${TEST_DATA_DIR}")

# Print summary
message(STATUS "AMGX found at: ${AMGX_ROOT_DIR}")
message(STATUS "CUDA found at: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "Building AMGX solver with C++17 support.")

######################
cmake_minimum_required(VERSION 3.10)
project(AMGXProject)

# Find CUDA since AMGX requires it
find_package(CUDA REQUIRED)

# Set AMGX paths relative to project root
set(AMGX_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/amgx")
set(AMGX_INCLUDE_DIR "${AMGX_ROOT_DIR}/include")
set(AMGX_LIBRARY_DIR "${AMGX_ROOT_DIR}/lib")

# Set test data path
set(TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test_data")

# Add include directories
include_directories(${AMGX_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})  # For solver directory

# Create library from AMGXSolver
add_library(solver
    solver/AMGXSolver.cpp
)

target_link_libraries(solver amgx)

# Add tests
add_executable(test_solver
    tests/test_solver.cpp
)

target_link_libraries(test_solver solver)

# Define test data path for the test executable
target_compile_definitions(test_solver PRIVATE 
    TEST_DATA_DIR="${TEST_DATA_DIR}") 