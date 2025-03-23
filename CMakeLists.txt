# Specify minimum CMake version and project name
cmake_minimum_required(VERSION 3.17)
project(AMGXProject)

# Set C++11 as required for all targets
# This is needed for modern C++ features and compatibility with CUDA
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set the AMGX library name based on the operating system
# Windows uses .lib/.dll, Linux uses .so, macOS is not supported
if(APPLE)
    message(FATAL_ERROR "AMGX is not supported on macOS. Compilation aborted.")
elseif(WIN32)
    set(AMGX_LIB_NAME "amgxsh.lib")
else()  # Default to Linux
    set(AMGX_LIB_NAME "libamgxsh.so")
endif()

# Find CUDA - required for AMGX which is a GPU-accelerated library
find_package(CUDAToolkit REQUIRED)

# Find Python and pybind11 - needed for creating Python bindings
# pybind11 is a library that makes it easy to create Python bindings for C++ code
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
# Get pybind11's CMake directory from Python to ensure we use the correct version
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED PATHS ${pybind11_DIR} NO_DEFAULT_PATH)

# Allow user to specify custom AMGX installation path
# If not specified, use the default path in extern/amgx
set(AMGX_CUSTOM_PATH "" CACHE PATH "Custom AMGX installation path")

if(AMGX_CUSTOM_PATH)
    set(AMGX_ROOT_DIR "${AMGX_CUSTOM_PATH}")
else()
    set(AMGX_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/amgx")
endif()

# Set paths for AMGX headers and libraries
set(AMGX_INCLUDE_DIR "${AMGX_ROOT_DIR}/include")

# Set library directory based on OS
# Windows and Linux use different directory structures
if(APPLE)
    message(FATAL_ERROR "AMGX is not supported on macOS. Compilation aborted.")
elseif(WIN32)
    set(AMGX_LIBRARY_DIR "${AMGX_ROOT_DIR}/lib")
else()  # Default to Linux
    set(AMGX_LIBRARY_DIR "${AMGX_ROOT_DIR}/build")
endif()

# Verify that AMGX is actually installed where we expect it
if(NOT EXISTS "${AMGX_INCLUDE_DIR}/amgx_c.h")
    message(FATAL_ERROR "AMGX header not found: ${AMGX_INCLUDE_DIR}/amgx_c.h. Set AMGX_CUSTOM_PATH correctly.")
endif()
if(NOT EXISTS "${AMGX_LIBRARY_DIR}/${AMGX_LIB_NAME}")
    message(FATAL_ERROR "AMGX library not found at ${AMGX_LIBRARY_DIR}/${AMGX_LIB_NAME}")
endif()

# Make AMGX headers available to all targets
include_directories(${AMGX_INCLUDE_DIR})

# Create our main solver library as a shared library (DLL on Windows)
add_library(solver SHARED ${CMAKE_CURRENT_SOURCE_DIR}/SRC/solver/AMGXSolver.cpp)
target_include_directories(solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/SRC/solver)
# Define BUILDING_SOLVER when building the DLL (needed for Windows DLL exports)
target_compile_definitions(solver PRIVATE BUILDING_SOLVER)

if(MSVC)
    # On Windows, we need to be explicit about where output files go
    # This ensures all DLLs end up in the same directory which is crucial for loading
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

    # Set the output directory for all build configurations (Debug, Release, etc.)
    foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR})
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR})
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR})
    endforeach()

    # Ensure the solver library goes to the right place
    set_target_properties(solver PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}
    )
endif()
# Link against AMGX library
target_link_libraries(solver PRIVATE ${AMGX_LIBRARY_DIR}/${AMGX_LIB_NAME})

# Create test executable that uses our solver library
add_executable(test_solver ${CMAKE_CURRENT_SOURCE_DIR}/SRC/tests/test_solver.cpp)
target_include_directories(test_solver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/SRC/solver)
target_link_libraries(test_solver PRIVATE solver)

# Create Python module using pybind11
pybind11_add_module(pyAMGXSolver ${CMAKE_CURRENT_SOURCE_DIR}/SRC/solver/pyAMGXSolver.cpp)
if(MSVC)
    # On Windows, ensure Python module goes to the same directory as other DLLs
    set_target_properties(pyAMGXSolver PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}
    )

    # Copy required DLLs to the output directory
    # This is crucial on Windows where DLLs must be in same directory or in PATH
    add_custom_command(TARGET pyAMGXSolver POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:solver>
            $<TARGET_FILE_DIR:pyAMGXSolver>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${AMGX_LIBRARY_DIR}/amgxsh.dll"
            ${CMAKE_BINARY_DIR}
    )
endif()
# Link Python module against our solver library
target_link_libraries(pyAMGXSolver PRIVATE solver)

# Print helpful information about what we found and what we're building
message(STATUS "AMGX found at: ${AMGX_ROOT_DIR}")
message(STATUS "CUDA found at: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "Building AMGX solver with C++11 support.")
