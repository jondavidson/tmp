import numpy as np
import polars as pl
from numba import njit

class NumpyDataConverter:
    def __init__(self, df: pl.DataFrame, group_start, group_end, upcast_types=True):
        """
        Initialize the class by converting a Polars DataFrame to NumPy 2D arrays
        and store group boundaries.
        
        Arguments:
        df -- Polars DataFrame with columns of various types.
        group_start -- List or array with start indices of groups.
        group_end -- List or array with end indices of groups.
        upcast_types -- Whether to upcast smaller types (e.g., float32 to float64, int32 to int64).
        """
        self.float_data, self.int_data = self._polars_to_numpy(df, upcast_types)
        self.group_start = group_start
        self.group_end = group_end

    def _polars_to_numpy(self, df: pl.DataFrame, upcast_types: bool):
        """
        Convert a Polars DataFrame into 2D NumPy arrays by data type, upcasting types as needed.
        
        Arguments:
        df -- Polars DataFrame with columns of various types.
        upcast_types -- Whether to upcast smaller types (e.g., float32 to float64, int32 to int64).
        
        Returns:
        float_data -- A 2D NumPy array containing all float-type columns.
        int_data -- A 2D NumPy array containing all integer-type columns.
        """
        float_cols = []
        int_cols = []

        for col in df.columns:
            series = df[col]
            np_array = series.to_numpy()

            if np.issubdtype(np_array.dtype, np.floating):
                if upcast_types and np_array.dtype == np.float32:
                    np_array = np_array.astype(np.float64)
                float_cols.append(np_array)
            elif np.issubdtype(np_array.dtype, np.integer):
                if upcast_types and np_array.dtype == np.int32:
                    np_array = np_array.astype(np.int64)
                int_cols.append(np_array)
        
        float_data = np.column_stack(float_cols) if float_cols else None
        int_data = np.column_stack(int_cols) if int_cols else None

        return float_data, int_data

    def get_group_chunk(self, group_id: int):
        """
        Access a chunk of the data corresponding to a specific group by slicing the 2D arrays.
        
        Arguments:
        group_id -- The ID of the group to extract.
        
        Returns:
        float_chunk -- The float data for the specified group.
        int_chunk -- The integer data for the specified group.
        """
        start_idx = self.group_start[group_id]
        end_idx = self.group_end[group_id]

        float_chunk = self.float_data[start_idx:end_idx, :] if self.float_data is not None else None
        int_chunk = self.int_data[start_idx:end_idx, :] if self.int_data is not None else None

        return float_chunk, int_chunk

    def process_group(self, group_id: int, func):
        """
        Apply the expensive Numba-compiled function to the specified group and return results.
        
        Arguments:
        group_id -- The ID of the group to process.
        func -- The Numba-compiled function to apply to the group's data.
        
        Returns:
        bool_output -- The Boolean vector result for the group.
        scalar_output -- The scalar results for the group (5 elements).
        """
        float_chunk, int_chunk = self.get_group_chunk(group_id)

        bool_output = np.empty(float_chunk.shape[0], dtype=np.bool_)
        scalar_output = np.zeros(5, dtype=np.float64)

# Example Polars DataFrame with data
df = pl.DataFrame({
    "float_col1": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
    "float_col2": [6.1, 7.2, 8.3, 9.4, 10.5, 11.6],
    "int_col1": [1, 2, 3, 4, 5, 6]
})

# Group indices stored separately
group_start = [0, 3]  # Start indices of the groups
group_end = [3, 6]    # End indices of the groups

# Initialize the converter with the Polars DataFrame and group boundaries
converter = NumpyDataConverter(df, group_start, group_end)

# Numba-compiled function (simplified for example)
@njit
def expensive_function(float_data, int_data, bool_output, scalar_output):
    n = float_data.shape[0]
    for i in range(n):
        bool_output[i] = float_data[i, 0] > int_data[i, 0]  # Example condition
    scalar_output[0] = np.sum(float_data[:, 0])

# Process the first group (group_id = 0)
bool_result, scalar_result = converter.process_group(group_id=0, func=expensive_function)

print("Boolean vector for Group 0:", bool_result)
print("Scalar outputs for Group 0:", scalar_result)


        func(float_chunk, int_chunk, bool_output, scalar_output)

        return bool_output, scalar_output
