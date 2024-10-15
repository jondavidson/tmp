import numpy as np
import polars as pl
from numba import njit, types
from numba.typed import Dict

def polars_to_numpy_dict(df: pl.DataFrame, upcast_types=True):
    """
    Convert a Polars DataFrame into a dictionary of NumPy arrays, separated by data type.
    """
    data_dict = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64[:],  # Placeholder, we'll adjust types dynamically
    )

    for col in df.columns:
        series = df[col]
        np_array = series.to_numpy()

        # Upcast types if necessary
        if upcast_types:
            if np_array.dtype == np.float32:
                np_array = np_array.astype(np.float64)
            elif np_array.dtype == np.int32:
                np_array = np_array.astype(np.int64)
            elif np_array.dtype == np.uint32:
                np_array = np_array.astype(np.uint64)

        # Store arrays in the dictionary with their column names
        data_dict[col] = np_array

    return data_dict

@njit
def process_data(data_dict, group_start, group_end, func):
    """
    Process data in groups using the provided Numba-compiled function.
    """
    num_groups = len(group_start)
    # Preallocate results
    bool_outputs = []
    scalar_outputs = []

    for group_id in range(num_groups):
        start_idx = group_start[group_id]
        end_idx = group_end[group_id]

        # Extract group data
        group_data = Dict.empty(
            key_type=types.unicode_type,
            value_type=types.float64[:],  # Adjust type as needed
        )
        for key in data_dict:
            group_data[key] = data_dict[key][start_idx:end_idx]

        # Initialize outputs for the group
        n = end_idx - start_idx
        bool_output = np.empty(n, dtype=np.bool_)
        scalar_output = np.zeros(5, dtype=np.float64)

        # Call the Numba-compiled function
        func(group_data, bool_output, scalar_output)

        # Accumulate results
        bool_outputs.append(bool_output)
        scalar_outputs.append(scalar_output)

    return bool_outputs, scalar_outputs

# Example Numba-compiled function
@njit
def expensive_function(group_data, bool_output, scalar_output):
    n = len(bool_output)
    float_col1 = group_data["float_col1"]
    int_col1 = group_data["int_col1"]
    for i in range(n):
        bool_output[i] = float_col1[i] > int_col1[i]
    scalar_output[0] = np.sum(float_col1)

# Example Polars DataFrame with data
df = pl.DataFrame({
    "float_col1": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
    "float_col2": [6.1, 7.2, 8.3, 9.4, 10.5, 11.6],
    "int_col1": [1, 2, 3, 4, 5, 6],
    "bool_col": [True, False, True, False, True, False],
    "uint_col": np.array([1, 2, 3, 4, 5, 6], dtype=np.uint32),
})

# Convert DataFrame to NumPy arrays in a dictionary
data_dict = polars_to_numpy_dict(df)

# Group indices
group_start = [0, 3]
group_end = [3, 6]

# Process data
bool_results, scalar_results = process_data(data_dict, group_start, group_end, expensive_function)

# Print results
for i, (bool_res, scalar_res) in enumerate(zip(bool_results, scalar_results)):
    print(f"Group {i} Boolean Output:", bool_res)
    print(f"Group {i} Scalar Output:", scalar_res)
