import numpy as np
import polars as pl
from numba import njit

class NumpyDataConverter:
    def __init__(self, df: pl.DataFrame, group_start, group_end, upcast_types=True):
        """
        Initialize the class by converting a Polars DataFrame to NumPy 2D arrays
        and store group boundaries.
        """
        self.float_data, self.int_data = self._polars_to_numpy(df, upcast_types)
        self.group_start = group_start
        self.group_end = group_end

    def _polars_to_numpy(self, df: pl.DataFrame, upcast_types: bool):
        """
        Convert a Polars DataFrame into 2D NumPy arrays by data type, upcasting types as needed.
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
        """
        start_idx = self.group_start[group_id]
        end_idx = self.group_end[group_id]

        float_chunk = self.float_data[start_idx:end_idx, :] if self.float_data is not None else None
        int_chunk = self.int_data[start_idx:end_idx, :] if self.int_data is not None else None

        return float_chunk, int_chunk

    def process_group(self, group_id: int, func):
        """
        Apply the expensive Numba-compiled function to the specified group and return results.
        """
        float_chunk, int_chunk = self.get_group_chunk(group_id)

        bool_output = np.empty(float_chunk.shape[0], dtype=np.bool_)
        scalar_output = np.zeros(5, dtype=np.float64)

        # Call the Numba-compiled function with the group data
        func(float_chunk, int_chunk, bool_output, scalar_output)

        # Return the computed outputs
        return bool_output, scalar_output

# Example Polars DataFrame with data
df = pl.DataFrame({
    "float_col1": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
    "float_col2": [6.1, 7.2, 8.3, 9.4, 10.5, 11.6],
    "int_col1": [1, 2, 3, 4, 5, 6]
})

# Group indices stored separately
group_start = [0, 3]  # Start
