# import re

# def parse_amgx_log(log_file):
#     """
#     Parses an AMGX log file to extract performance metrics.

#     Returns:
#         dict: A dictionary containing:
#               - total_iterations
#               - avg_convergence_rate
#               - final_residual
#               - total_reduction
#               - max_memory_usage
#               - total_time
#               - setup_time
#               - solve_time
#               - solve_time_per_iteration
#               - solver_status
#     """
#     data = {
#         "total_iterations": None,
#         "avg_convergence_rate": None,
#         "final_residual": None,
#         "total_reduction": None,
#         "max_memory_usage": None,
#         "total_time": None,
#         "setup_time": None,
#         "solve_time": None,
#         "solve_time_per_iteration": None,
#         "solver_status": None,
#     }

#     try:
#         with open(log_file, "r") as f:
#             log_content = f.read()

#         # Use re.findall() to get all matches and pick the last one
#         def find_last(pattern, text, cast_func=float):
#             matches = re.findall(pattern, text)
#             return cast_func(matches[-1]) if matches else None

#         data["total_iterations"] = find_last(r"Total Iterations:\s+(\d+)", log_content, int)
#         data["avg_convergence_rate"] = find_last(r"Avg Convergence Rate:\s+([\d.e+-]+)", log_content)
#         data["final_residual"] = find_last(r"Final Residual:\s+([\d.e+-]+)", log_content)
#         data["total_reduction"] = find_last(r"Total Reduction in Residual:\s+([\d.e+-]+)", log_content)
#         data["max_memory_usage"] = find_last(r"Maximum Memory Usage:\s+([\d.e+-]+) GB", log_content)
#         data["total_time"] = find_last(r"Total Time:\s+([\d.e+-]+)", log_content)
#         data["setup_time"] = find_last(r"setup:\s+([\d.e+-]+)", log_content)
#         data["solve_time"] = find_last(r"solve:\s+([\d.e+-]+)", log_content)
#         data["solve_time_per_iteration"] = find_last(r"solve\(per iteration\):\s+([\d.e+-]+)", log_content)
#         data["solver_status"] = find_last(r"Solver Status:\s+(-?\d+)", log_content, int)

#     except Exception as e:
#         print(f"[ERROR] Failed to parse log {log_file}: {e}")

#     return data

# # Test the parser separately
# if __name__ == "__main__":
#     test_log_file = "matrix_tests/logs/sample_matrix.log"
#     parsed_data = parse_amgx_log(test_log_file)
#     print(parsed_data)

import re

def parse_amgx_log(log_file):
    """
    Parses an AMGX log file to extract performance metrics.
    
    Handles missing values (e.g., "-"), NaN values, and conversion errors.
    
    Returns:
        dict: A dictionary containing:
              - total_iterations
              - avg_convergence_rate
              - final_residual
              - total_reduction
              - max_memory_usage
              - total_time
              - setup_time
              - solve_time
              - solve_time_per_iteration
              - solver_status
    """
    data = {
        "total_iterations": None,
        "avg_convergence_rate": None,
        "final_residual": None,
        "total_reduction": None,
        "max_memory_usage": None,
        "total_time": None,
        "setup_time": None,
        "solve_time": None,
        "solve_time_per_iteration": None,
        "solver_status": None,
    }

    try:
        with open(log_file, "r") as f:
            log_content = f.read()

        # Function to get last match, handle missing values and NaN cases
        def find_last(pattern, text, cast_func=float):
            matches = re.findall(pattern, text)
            if matches:
                last_value = matches[-1].strip().lower()  # Get last match, strip spaces, make lowercase
                if last_value in ("nan", "-nan", "-"):  # Handle NaN or missing cases
                    return None  # Or return float('nan') if you prefer
                try:
                    return cast_func(last_value)
                except ValueError:
                    return None  # Ensure no crashes due to conversion errors
            return None

        # Extract values from log
        data["total_iterations"] = find_last(r"Total Iterations:\s+(\d+)", log_content, int)
        data["avg_convergence_rate"] = find_last(r"Avg Convergence Rate:\s+([\d.e+-]+|-|nan|-nan)", log_content)
        data["final_residual"] = find_last(r"Final Residual:\s+([\d.e+-]+|-|nan|-nan)", log_content)
        data["total_reduction"] = find_last(r"Total Reduction in Residual:\s+([\d.e+-]+|-|nan|-nan)", log_content)
        data["max_memory_usage"] = find_last(r"Maximum Memory Usage:\s+([\d.e+-]+|-|nan|-nan) GB", log_content)
        data["total_time"] = find_last(r"Total Time:\s+([\d.e+-]+|-|nan|-nan)", log_content)
        data["setup_time"] = find_last(r"setup:\s+([\d.e+-]+|-|nan|-nan)", log_content)
        data["solve_time"] = find_last(r"solve:\s+([\d.e+-]+|-|nan|-nan)", log_content)
        data["solve_time_per_iteration"] = find_last(r"solve\(per iteration\):\s+([\d.e+-]+|-|nan|-nan)", log_content)
        data["solver_status"] = find_last(r"Solver Status:\s+(-?\d+)", log_content, int)

    except Exception as e:
        print(f"[ERROR] Failed to parse log {log_file}: {e}")

    return data

# Test the parser separately
if __name__ == "__main__":
    test_log_file = "matrix_tests/logs/sample_matrix.log"
    parsed_data = parse_amgx_log(test_log_file)
    print(parsed_data)

