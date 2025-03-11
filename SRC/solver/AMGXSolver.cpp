#include "AMGXSolver.h"

/* 
Note from AMGX Reference:
It is recommended that the host buffers passed to AMGX vector upload be pinned 
previously via AMGX pin memory. This allows the underlying CUDA driver to 
achieve higher data transfer rates across the PCI-Express bus.*/
AMGXSolver::AMGXSolver(const char *config_file, bool use_cpu, const int *gpu_ids, 
                        int num_gpus, bool pin_memory)
    : _use_cpu(use_cpu), _pin_memory(pin_memory)
{
    if (config_file == nullptr) {
        throw std::invalid_argument("Configuration file path cannot be null");
    }

    if (!_use_cpu && gpu_ids == nullptr) {
        throw std::invalid_argument("GPU IDs array cannot be null when using GPU mode");
    }

    if (!_use_cpu && num_gpus <= 0) {
        throw std::invalid_argument("Number of GPUs must be positive when using GPU mode");
    }

    if (_use_cpu && gpu_ids != nullptr)
    {
        std::cout << "Cannot specify both CPU mode and GPU IDs." 
                  << "GPU IDs will be ignored.\n";
    }

    /* Initialize AMGX library with user-defined error handling*/
    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_initialize_plugins());
    AMGX_SAFE_CALL(AMGX_register_print_callback(&AMGXSolver::callback));
    AMGX_SAFE_CALL(AMGX_install_signal_handler());

    /* AMGX configuration file*/
    AMGX_SAFE_CALL(AMGX_config_create_from_file(&_config, config_file));

    /* AMGX mode and resources*/
    if (_use_cpu)
    {
        _mode = AMGX_mode_hDDI;
        AMGX_SAFE_CALL(AMGX_resources_create_simple(&_resources, _config));
    }
    else
    {
        _mode = AMGX_mode_dDDI;
        AMGX_SAFE_CALL(AMGX_resources_create(&_resources, _config, NULL, num_gpus, 
                                                gpu_ids));
    }

    /* AMGX matrices and vectors */
    AMGX_SAFE_CALL(AMGX_matrix_create(&_matrix, _resources, _mode));
    AMGX_SAFE_CALL(AMGX_vector_create(&_rhs, _resources, _mode));
    AMGX_SAFE_CALL(AMGX_vector_create(&_solution, _resources, _mode));

    /* AMGX solver*/
    AMGX_SAFE_CALL(AMGX_solver_create(&_solver, _resources, _mode, _config));
}

AMGXSolver::~AMGXSolver()
{
    if (_solver) AMGX_solver_destroy(_solver);
    if (_solution) AMGX_vector_destroy(_solution);
    if (_rhs) AMGX_vector_destroy(_rhs);
    if (_matrix) AMGX_matrix_destroy(_matrix);
    if (_resources) AMGX_resources_destroy(_resources);
    if (_config) AMGX_config_destroy(_config);
    AMGX_finalize_plugins();
    AMGX_finalize();
}

void AMGXSolver::initializeMatrix(int num_rows, const int *row_ptr, 
    const int *col_indices, const double *values)
{   
    if (num_rows <= 0) {
        throw std::invalid_argument("Number of rows must be positive");
    }
    if (row_ptr == nullptr || col_indices == nullptr || values == nullptr) {
        throw std::invalid_argument("Matrix arrays cannot be null");
    }
    
    int num_non_zeros = row_ptr[num_rows];
    if (num_non_zeros <= 0) {
        throw std::invalid_argument("Number of non-zero elements must be positive");
    }
    
    if (!_use_cpu && _pin_memory) {
        AMGX_SAFE_CALL(AMGX_pin_memory((void*)row_ptr, sizeof(int) * num_non_zeros));
        AMGX_SAFE_CALL(AMGX_pin_memory((void*)col_indices, sizeof(int) * num_non_zeros));
        AMGX_SAFE_CALL(AMGX_pin_memory((void*)values, sizeof(double) * num_non_zeros));
    }
    AMGX_SAFE_CALL(AMGX_matrix_upload_all(_matrix, num_rows, num_non_zeros, 1, 1, 
                                        row_ptr, col_indices, values, nullptr));
    if (!_use_cpu && _pin_memory) {
        AMGX_SAFE_CALL(AMGX_unpin_memory((void*)row_ptr));
        AMGX_SAFE_CALL(AMGX_unpin_memory((void*)col_indices));
        AMGX_SAFE_CALL(AMGX_unpin_memory((void*)values));
    }
    AMGX_SAFE_CALL(AMGX_solver_setup(_solver, _matrix));
}

void AMGXSolver::replaceCoefficients(int num_rows, int num_non_zeros, 
                    const double *values)
{
    if (num_rows <= 0) {
        throw std::invalid_argument("Number of rows must be positive");
    }
    if (num_non_zeros <= 0) {
        throw std::invalid_argument("Number of non-zero elements must be positive");
    }
    if (values == nullptr) {
        throw std::invalid_argument("Values array cannot be null");
    }
    
    if (!_use_cpu && _pin_memory) {
        AMGX_SAFE_CALL(AMGX_pin_memory((void*)values, sizeof(double) * num_non_zeros));
    }
    AMGX_SAFE_CALL(AMGX_matrix_replace_coefficients(_matrix, num_rows, 
                                            num_non_zeros, values, nullptr));
    if (!_use_cpu && _pin_memory) {
        AMGX_SAFE_CALL(AMGX_unpin_memory((void*)values));
    }
    AMGX_SAFE_CALL(AMGX_solver_setup(_solver, _matrix));
}

int AMGXSolver::solve(void) 
{
    std::cerr << "solve(void) not implemented. Use solve(x, b, num_rows) instead.\n";
    return -1; // not implemented
}

int AMGXSolver::solve(double* x, const double* b, int num_rows) 
{
    if (num_rows <= 0) {
        throw std::invalid_argument("Number of rows must be positive");
    }
    if (x == nullptr || b == nullptr) {
        throw std::invalid_argument("Solution and RHS vectors cannot be null");
    }
    
    if (!_use_cpu && _pin_memory) {
        AMGX_SAFE_CALL(AMGX_pin_memory((void*)b, sizeof(double) * num_rows));
        AMGX_SAFE_CALL(AMGX_pin_memory((void*)x, sizeof(double) * num_rows));
    }
    AMGX_SAFE_CALL(AMGX_vector_upload(_rhs, num_rows, 1, b));
    
    AMGX_SAFE_CALL(AMGX_vector_set_zero(_solution, num_rows, 1));
    // slight optimization to tell it to start with solution being all zeros
    AMGX_SAFE_CALL(AMGX_solver_solve_with_0_initial_guess(_solver, _rhs, _solution));
    
    
    AMGX_SAFE_CALL(AMGX_vector_download(_solution, x));
    if (!_use_cpu && _pin_memory) {
        AMGX_SAFE_CALL(AMGX_unpin_memory((void*)x));
        AMGX_SAFE_CALL(AMGX_unpin_memory((void*)b));
    }
    /* AMGX check status */
    AMGX_SOLVE_STATUS status;
    AMGX_SAFE_CALL(AMGX_solver_get_status(_solver, &status));

    switch (status) {
        case AMGX_SOLVE_SUCCESS:
            return 0;
        case AMGX_SOLVE_FAILED:
            std::cerr << "[Error] AMGX solve failed.\n";
            return -1;
        case AMGX_SOLVE_DIVERGED:
            std::cerr << "[Error] AMGX solve did not converge.\n";
            return -2;
        default:
            std::cerr << "[Error] AMGX solver returned an unknown status.\n";
            return -3;
    }
}

/* print callback (could be customized) */
void AMGXSolver::callback(const char* msg, int length) 
{
    std::cerr << msg << "\n";
}
