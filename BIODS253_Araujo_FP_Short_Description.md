# Final Project Proposal: Accelerating OpenSees with GPU-Based Solvers
- Author: Gustavo A. Araújo R.
- Email: garaujor@stanford.edu

## Overview
For my final project, I aim to extend the OpenSees framework to solve systems of linear equations using GPUs for rendering large-scale structural simulations more computationally-tractable compared to CPU-only solvers.

OpenSees (Open System for Earthquake Engineering Simulation) is an open-source software framework written in C++ with a Python interpreter. It is widely used in civil engineering to simulate structural and geotechnical systems under various loading conditions, including earthquakes. The [OpenSees Source Code](https://github.com/OpenSees/OpenSees) is available on GitHub, with its Python Interpreter documented in the [OpenSeesPy Documentation](https://openseespydoc.readthedocs.io/en/latest/).

Efforts to introduce GPU acceleration into OpenSees have been made in the past, such as the integration of the Cusp library ([OpenSees Cusp Integration](https://opensees.berkeley.edu/wiki/index.php/Cusp)). However, Cusp has since become obsolete, and other libraries that leverage modern GPU hardware provide new opportunities for GPU acceleration. For this project, I plan to leverage two CUDA-based libraries developed by NVIDIA:
- **AmgX**: A library designed specifically for solving large sparse linear systems with algebraic multigrid methods.
- **CUSOLVER**: A more general-purpose library for dense and sparse linear algebra operations.

By integrating these modern libraries, the project will bring GPU-based solvers back to OpenSees.

## Objectives
1. Implement GPU-accelerated solvers in OpenSees using AmgX and/or CUSOLVER.
2. Benchmark the performance of GPU-based solvers against traditional CPU solvers for a set of structural models of varying size and complexity.