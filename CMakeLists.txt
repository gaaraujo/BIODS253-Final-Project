cmake_minimum_required(VERSION 3.17)
project(AMGXProject)

# Require C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find CUDA (modern method)
find_package(CUDAToolkit REQUIRED)

# Allow user to specify a custom AMGX installation path
set(AMGX_CUSTOM_PATH "" CACHE PATH "Custom AMGX installation path")

# Set AMGX paths: use custom path if specified, otherwise use the default
if(AMGX_CUSTOM_PATH)
    set(AMGX_ROOT_DIR "${AMGX_CUSTOM_PATH}")
else()
    set(AMGX_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/amgx")
endif()

set(AMGX_INCLUDE_DIR "${AMGX_ROOT_DIR}/include")
set(AMGX_LIBRARY_DIR "${AMGX_ROOT_DIR}/lib")

# Ensure AMGX headers exist
if(NOT EXISTS "${AMGX_INCLUDE_DIR}/amgx_c.h")
    message(FATAL_ERROR "AMGX header not found: ${AMGX_INCLUDE_DIR}/amgx_c.h. Set AMGX_CUSTOM_PATH correctly.")
endif()

# Ensure AMGX library exists
if(NOT EXISTS "${AMGX_LIBRARY_DIR}/libamgxsh.so")
    message(FATAL_ERROR "AMGX library not found at ${AMGX_LIBRARY_DIR}/libamgxsh.so")
endif()

# Include AMGX headers
include_directories(${AMGX_INCLUDE_DIR})

# Create library from AMGXSolver
add_library(solver ${CMAKE_CURRENT_SOURCE_DIR}/SRC/solver/AMGXSolver.cpp)

# Ensure solver can find its own headers
target_include_directories(solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/SRC/solver)

# Link AMGX to solver
target_link_libraries(solver PRIVATE ${AMGX_LIBRARY_DIR}/libamgxsh.so)

# Add tests executable
add_executable(test_solver ${CMAKE_CURRENT_SOURCE_DIR}/SRC/tests/test_solver.cpp)

# Ensure test_solver can find solver headers
target_include_directories(test_solver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/SRC/solver)

# Link test_solver with solver
target_link_libraries(test_solver PRIVATE solver)

# Print summary
message(STATUS "AMGX found at: ${AMGX_ROOT_DIR}")
message(STATUS "CUDA found at: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "Building AMGX solver with C++11 support.")
