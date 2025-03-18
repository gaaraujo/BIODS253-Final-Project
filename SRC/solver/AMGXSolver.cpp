#include "AMGXSolver.h"

std::ofstream AMGXSolver::_log_file_stream;
bool AMGXSolver::_use_log_file = false;
bool AMGXSolver::_amgx_initialized = false;
int AMGXSolver::_active_solver_instances = 0; // Track active instances

/* 
Note from AMGX Reference:
It is recommended that the host buffers passed to AMGX vector upload be pinned 
previously via AMGX pin memory. This allows the underlying CUDA driver to 
achieve higher data transfer rates across the PCI-Express bus.*/

AMGXSolver::AMGXSolver(const char *config_file, bool use_cpu, const int *gpu_ids, 
                        int num_gpus, bool pin_memory, const char* log_file)
{
    try {
        initialize(config_file, use_cpu, gpu_ids, num_gpus, pin_memory, log_file);
    }
    catch (const std::exception &e) {
        std::cerr << "[ERROR] AMGXSolver constructor failed: " << e.what() << std::endl;
        throw; // Rethrow exception to inform the caller
    }
}


AMGXSolver::~AMGXSolver()
{
    try {
        cleanup();
    } 
    catch (const std::exception &e) {
        std::cerr << "[ERROR] Exception in AMGXSolver destructor: " << e.what() << std::endl;
    }
}


void AMGXSolver::initialize(const char *config_file, bool use_cpu, const int *gpu_ids, 
                            int num_gpus, bool pin_memory, const char* log_file) 
{
    try {
        if (config_file == nullptr) {
            throw std::invalid_argument("Configuration file path cannot be null");
        }

        if (!use_cpu && gpu_ids == nullptr) {
            throw std::invalid_argument("GPU IDs array cannot be null when using GPU mode");
        }

        if (!use_cpu && num_gpus <= 0) {
            throw std::invalid_argument("Number of GPUs must be positive when using GPU mode");
        }

        if (use_cpu && gpu_ids != nullptr) {
            std::cout << "[WARNING] CPU mode selected, ignoring provided GPU IDs.\n";
        }

        // Open log file if provided
        if (log_file != nullptr) {
            _log_file_stream.open(log_file, std::ios::out | std::ios::app);
            if (!_log_file_stream.is_open()) {
                throw std::runtime_error("Failed to open log file.");
            }
            _use_log_file = true;
        } else {
            _use_log_file = false;
        }

        /* Initialize AMGX library */
        if (!_amgx_initialized) {
            CHECK_AMGX_CALL(AMGX_initialize());
            CHECK_AMGX_CALL(AMGX_register_print_callback(&AMGXSolver::callback));
            CHECK_AMGX_CALL(AMGX_install_signal_handler());
            _amgx_initialized = true;
        }

        /* Create AMGX configuration */
        if (_config != nullptr) {
            AMGX_config_destroy(_config);
            _config = nullptr;
        }
        CHECK_AMGX_CALL(AMGX_config_create_from_file(&_config, config_file));

        // This doesn't work and I can't figure out why
        // **Override configuration parameters for residual tracking**
        // CHECK_AMGX_CALL(AMGX_config_add_parameters(&_config, "monitor_residual=1, store_res_history=1"));

        /* Set AMGX mode and create resources */
        _use_cpu = use_cpu;
        _pin_memory = pin_memory;
        if (_resources != nullptr) {
            AMGX_resources_destroy(_resources);
            _resources = nullptr;
        }
        if (_use_cpu) {
            _mode = AMGX_mode_hDDI;
            CHECK_AMGX_CALL(AMGX_resources_create_simple(&_resources, _config));
        } else {
            _mode = AMGX_mode_dDDI;
            CHECK_AMGX_CALL(AMGX_resources_create(&_resources, _config, nullptr, num_gpus, gpu_ids));
        }

        /* Create solver */
        if (_solver != nullptr) {
            AMGX_solver_destroy(_solver);
            _solver = nullptr;
        }
        CHECK_AMGX_CALL(AMGX_solver_create(&_solver, _resources, _mode, _config));

        /* Create matrix and vectors */
        if (_matrix != nullptr) {
            AMGX_matrix_destroy(_matrix);
            _matrix = nullptr;
        }
        if (_rhs != nullptr) {
            AMGX_vector_destroy(_rhs);
            _rhs = nullptr;
        }
        if (_solution != nullptr) {
            AMGX_vector_destroy(_solution);
            _solution = nullptr;
        }
        CHECK_AMGX_CALL(AMGX_matrix_create(&_matrix, _resources, _mode));
        CHECK_AMGX_CALL(AMGX_vector_create(&_rhs, _resources, _mode));
        CHECK_AMGX_CALL(AMGX_vector_create(&_solution, _resources, _mode));

        _active_solver_instances++;
    }
    catch (const std::exception &e) {
        std::cerr << "[ERROR] AMGXSolver initialization failed: " << e.what() << std::endl;
        cleanup(); // Ensure cleanup on failure
        throw; // Re-throw exception to notify caller
    }
}


void AMGXSolver::cleanup() {
    try {
        if (_solution) { AMGX_vector_destroy(_solution); _solution = nullptr; }
        if (_rhs) { AMGX_vector_destroy(_rhs); _rhs = nullptr; }
        if (_matrix) { AMGX_matrix_destroy(_matrix); _matrix = nullptr; }
        if (_solver) { AMGX_solver_destroy(_solver); _solver = nullptr; }
        if (_resources) { AMGX_resources_destroy(_resources); _resources = nullptr; }
        if (_config) { AMGX_config_destroy(_config); _config = nullptr; }

        if (_active_solver_instances > 0) {
            _active_solver_instances--;
        }

        // Finalize AMGX only when last instance is destroyed
        if (_active_solver_instances == 0 && _amgx_initialized) {
            AMGX_reset_signal_handler();
            AMGX_finalize();
            _amgx_initialized = false;

            if (_use_log_file && _log_file_stream.is_open()) {
                _log_file_stream.close();
            }
        }
    } 
    catch (const std::exception &e) {
        std::cerr << "[ERROR] Exception in AMGXSolver::cleanup: " << e.what() << std::endl;
    }
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
    
    if (row_ptr[num_rows] <= 0) {
        throw std::invalid_argument("Number of non-zero elements must be positive");
    }
    
    this->_num_rows = num_rows;
    this->_num_non_zeros = row_ptr[num_rows];

    if (!_use_cpu && _pin_memory) {
        CHECK_AMGX_CALL(AMGX_pin_memory((void*)row_ptr, sizeof(int) * (_num_rows + 1)));
        CHECK_AMGX_CALL(AMGX_pin_memory((void*)col_indices, sizeof(int) * _num_non_zeros));
        CHECK_AMGX_CALL(AMGX_pin_memory((void*)values, sizeof(double) * _num_non_zeros));
    }
    CHECK_AMGX_CALL(AMGX_matrix_upload_all(_matrix, _num_rows, _num_non_zeros, 1, 1, 
                                        row_ptr, col_indices, values, nullptr));
    if (!_use_cpu && _pin_memory) {
        CHECK_AMGX_CALL(AMGX_unpin_memory((void*)row_ptr));
        CHECK_AMGX_CALL(AMGX_unpin_memory((void*)col_indices));
        CHECK_AMGX_CALL(AMGX_unpin_memory((void*)values));
    }
    CHECK_AMGX_CALL(AMGX_solver_setup(_solver, _matrix));
}

void AMGXSolver::replaceCoefficients(int num_rows, int num_non_zeros, 
                    const double *values)
{
    if (this->_num_non_zeros <= 0 || this->_num_rows <= 0) {
        throw std::invalid_argument("Matrix has not been initialized. Use AMGXSolver::initializeMatrix() method instead");
    }

    if (num_rows <= 0) {
        throw std::invalid_argument("Number of rows must be positive");
    }
    if (num_non_zeros <= 0) {
        throw std::invalid_argument("Number of non-zero elements must be positive");
    }
    if (values == nullptr) {
        throw std::invalid_argument("Values array cannot be null");
    }
    
    if (num_rows != this->_num_rows) {
        throw std::invalid_argument("Number of rows must match matrix size. Use AMGXSolver::initializeMatrix() method instead");
    }
    if (num_non_zeros != this->_num_non_zeros) {
        throw std::invalid_argument("Number of rows must match matrix size. Use AMGXSolver::initializeMatrix() method instead");
    }
    
    if (!_use_cpu && _pin_memory) {
        CHECK_AMGX_CALL(AMGX_pin_memory((void*)values, sizeof(double) * _num_non_zeros));
    }

    CHECK_AMGX_CALL(AMGX_matrix_replace_coefficients(_matrix, _num_rows, 
                                            _num_non_zeros, values, nullptr));
    if (!_use_cpu && _pin_memory) {
        CHECK_AMGX_CALL(AMGX_unpin_memory((void*)values));
    }
    CHECK_AMGX_CALL(AMGX_solver_setup(_solver, _matrix));
}

int AMGXSolver::solve(void) 
{
    std::cerr << "solve(void) not implemented. Use solve(x, b, num_rows) instead.\n";
    return -1; // not implemented
}

int AMGXSolver::solve(double* x, const double* b, int num_rows) 
{
    if (this->_num_non_zeros <= 0 || this->_num_rows <= 0) {
        throw std::invalid_argument("Matrix has not been initialized. Use AMGXSolver::initializeMatrix()");
    }

    if (num_rows <= 0) {
        throw std::invalid_argument("Number of rows must be positive");
    }
    if (x == nullptr || b == nullptr) {
        throw std::invalid_argument("Solution and RHS vectors cannot be null");
    }
    
    if (num_rows != this->_num_rows) {
        throw std::invalid_argument("Number of rows must match matrix size");
    }

    if (!_use_cpu && _pin_memory) {
        CHECK_AMGX_CALL(AMGX_pin_memory((void*)b, sizeof(double) * num_rows));
        CHECK_AMGX_CALL(AMGX_pin_memory((void*)x, sizeof(double) * num_rows));
    }
    CHECK_AMGX_CALL(AMGX_vector_upload(_rhs, num_rows, 1, b));
    
    CHECK_AMGX_CALL(AMGX_vector_set_zero(_solution, num_rows, 1));
    // slight optimization to tell it to start with solution being all zeros
    CHECK_AMGX_CALL(AMGX_solver_solve_with_0_initial_guess(_solver, _rhs, _solution));
    
    
    CHECK_AMGX_CALL(AMGX_vector_download(_solution, x));
    if (!_use_cpu && _pin_memory) {
        CHECK_AMGX_CALL(AMGX_unpin_memory((void*)x));
        CHECK_AMGX_CALL(AMGX_unpin_memory((void*)b));
    }
    /* AMGX check status */
    AMGX_SOLVE_STATUS status;
    CHECK_AMGX_CALL(AMGX_solver_get_status(_solver, &status));
    
    char status_message[50]; 
    snprintf(status_message, sizeof(status_message), "Solver Status: %d\n", -status);
    callback(status_message, strlen(status_message));
    
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

int AMGXSolver::getNumIterations(void) {
    int n;
    AMGX_solver_get_iterations_number(_solver, &n);
    return n;
}

double AMGXSolver::getFinalResidual(void) {
    double final_residual;
    AMGX_solver_get_iteration_residual(_solver, this->getNumIterations(), 0, &final_residual);
    return final_residual;
}

/* print callback (could be customized) */
void AMGXSolver::callback(const char* msg, int length) 
{
    if (_use_log_file && _log_file_stream.is_open()) {
        _log_file_stream.write(msg, length);
        _log_file_stream << std::endl;
    } else {
        std::cout.write(msg, length);
        std::cout << std::endl;
    }
}
