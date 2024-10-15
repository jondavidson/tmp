import numpy as np
import polars as pl
from numba import njit

class NumpyDataConverter:
    def __init__(self, df: pl.DataFrame, upcast_types=True):
        """
        Initialize the class by converting a Polars DataFrame to NumPy 2D arrays.
        
        Arguments:
        df -- Polars DataFrame with columns of various types.
        upcast_types -- Whether to upcast smaller types (e.g., float32 to float64, int32 to int64).
        """
        self.float_data, self.int_data, self.group_index = self._polars_to_numpy(df, upcast_types)

    def _polars_to_numpy(self, df: pl.DataFrame, upcast_types: bool):
        """
        Convert a Polars DataFrame into 2D NumPy arrays by data type, upcasting types as needed.
        
        Arguments:
        df -- Polars DataFrame with columns of various types.
        upcast_types -- Whether to upcast smaller types (e.g., float32 to float64, int32 to int64).
        
        Returns:
        float_data -- A 2D NumPy array containing all float-type columns.
        int_data -- A 2D NumPy array containing all integer-type columns.
        group_index -- A 2D NumPy array with the group start and end indices.
        """
        float_cols = []
        int_cols = []

        # Loop through each column in the Polars DataFrame
        for col in df.columns:
            series = df[col]
            np_array = series.to_numpy()

            # Determine if the column is numeric, and upcast if needed
            if np.issubdtype(np_array.dtype, np.floating):
                if upcast_types and np_array.dtype == np.float32:
                    np_array = np_array.astype(np.float64)
                float_cols.append(np_array)
            elif np.issubdtype(np_array.dtype, np.integer):
                if upcast_types and np_array.dtype == np.int32:
                    np_array = np_array.astype(np.int64)
                int_cols.append(np_array)
        
        # Stack the columns of the same type into 2D arrays
        float_data = np.column_stack(float_cols) if float_cols else None
        int_data = np.column_stack(int_cols) if int_cols else None

        # Assume group indices are provided in 'group_start' and 'group_end' columns
        group_index = np.column_stack([df['group_start'].to_numpy(), df['group_end'].to_numpy()])

        return float_data, int_data, group_index

    def get_group_chunk(self, group_id: int):
        """
        Access a chunk of the data corresponding to a specific group by slicing the 2D arrays.
        
        Arguments:
        group_id -- The ID of the group to extract.
        
        Returns:
        float_chunk -- The float data for the specified group.
        int_chunk -- The integer data for the specified group.
        """
        # Extract the start and end indices for the given group_id
        start_idx = self.group_index[group_id, 0]
        end_idx = self.group_index[group_id, 1]
        
        # Slice the 2D arrays to get the chunk for the group
        float_chunk = self.float_data[start_idx:end_idx, :] if self.float_data is not None else None
        int_chunk = self.int_data[start_idx:end_idx, :] if self.int_data is not None else None
        
        return float_chunk, int_chunk

    def process_group(self, group_id: int, func, bool_output, scalar_output):
        """
        Apply the expensive Numba-compiled function to the specified group.
        
        Arguments:
        group_id -- The ID of the group to process.
        func -- The Numba-compiled function to apply to the group's data.
        bool_output -- Preallocated Boolean array to store the Boolean vector result.
        scalar_output -- Preallocated array for storing scalar results (length = 5).
        
        Returns:
        result -- The result of the Numba-compiled function for the specified group.
        """
        # Get the group chunk
        float_chunk, int_chunk = self.get_group_chunk(group_id)

        # Call the Numba-compiled function with the chunked data
        func(float_chunk, int_chunk, bool_output, scalar_output)

# Example Numba function
@njit
def expensive_function(float_data, int_data, bool_output, scalar_output):
    n = float_data.shape[0]  # Number of rows in the chunk
    
    # Example Boolean vector computation
    for i in range(n):
        bool_output[i] = float_data[i, 0] > int_data[i, 0]  # Example condition
    
    # Example scalar computations (accumulating results)
    scalar_output[0] = np.sum(float_data[:, 0])  # Sum of the first float column
    scalar_output[1] = np.mean(float_data[:, 0])  # Mean of the first float column
    scalar_output[2] = np.sum(int_data[:, 0])  # Sum of the first int column
    scalar_output[3] = np.mean(int_data[:, 0])  # Mean of the first int column
    scalar_output[4] = np.sum(bool_output)  # Sum of True values in the Boolean vector

# Example usage of the class
df = pl.DataFrame({
    "float_col1": [1.1, 2.2, 3.3, 4.4, 5.5],
    "float_col2": [6.1, 7.2, 8.3, 9.4, 10.5],
    "int_col1": [1, 2, 3, 4, 5],
    "group_start": [0, 2],  # Start indices of the groups
    "group_end": [2, 5]     # End indices of the groups
})

# Initialize the converter with the Polars DataFrame
converter = NumpyDataConverter(df)

# Preallocate space for the outputs
bool_output = np.empty(3, dtype=np.bool_)  # Boolean vector for the group (length matches data chunk)
scalar_output = np.zeros(5, dtype=np.float64)  # Array to store 5 scalar results

# Process the first group (group_id = 0) using the Numba-compiled function
converter.process_group(group_id=0, func=expensive_function, bool_output=bool_output, scalar_output=scalar_output)

# Output the preallocated results
print("Boolean vector:", bool_output)
print("Scalar outputs:", scalar_output)
