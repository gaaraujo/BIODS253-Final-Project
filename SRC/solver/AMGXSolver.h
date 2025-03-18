#ifndef AMGX_h
#define AMGX_h

#include <amgx_c.h>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cstring>

// Macro to handle AMGX function calls and throw exceptions on errors
#define CHECK_AMGX_CALL(rc) \
{ \
  AMGX_RC err;     \
  char msg[4096];   \
  switch(err = (rc)) {    \
  case AMGX_RC_OK: \
    break; \
  default: \
    AMGX_get_error_string(err, msg, 4096); \
    fprintf(stderr, "AMGX ERROR in function: %s at file %s line %6d\n", #rc, __FILE__, __LINE__); \
    fprintf(stderr, "AMGX ERROR: %s\n", msg); \
    throw std::runtime_error(std::string("AMGX ERROR: ") + msg); \
  } \
}

class AMGXSolver
{
public:
    AMGXSolver(const char* config_file, bool use_cpu = false, const int* gpu_ids = nullptr,
               int num_gpus = 0, bool pin_memory = false, const char* log_file = nullptr);
    ~AMGXSolver();
    
    void initialize(const char *config_file, bool use_cpu, const int *gpu_ids, 
                            int num_gpus, bool pin_memory, const char* log_file); 
    void cleanup();

    void initializeMatrix(int num_rows, const int* row_ptr, const int* col_indices,
                   const double* values);
    void replaceCoefficients(int num_rows, int num_non_zeros, 
                    const double *values);
    int solve(void);
    int solve(double* x, const double* b, int num_rows);

private:
    static void callback(const char* msg, int length);
    static std::ofstream _log_file_stream;
    static bool _use_log_file;
    static bool _amgx_initialized;
    static int _active_solver_instances;

    AMGX_config_handle    _config       = nullptr;
    AMGX_resources_handle _resources    = nullptr;
    AMGX_matrix_handle    _matrix       = nullptr;
    AMGX_vector_handle    _rhs          = nullptr;
    AMGX_vector_handle    _solution     = nullptr;
    AMGX_solver_handle    _solver       = nullptr;
    AMGX_Mode             _mode;

    bool _use_cpu;
    bool _pin_memory;
    
    int _num_rows = 0;
    int _num_non_zeros = 0;
};
#endif