import numpy as np
import polars as pl
from numba import njit, types
from numba.typed import List

class DataProcessor:
    def __init__(self, df, group_start, group_end, num_epochs):
        self.df = df
        self.group_start = group_start
        self.group_end = group_end
        self.num_epochs = num_epochs

        # Prepare data arrays
        self.float_data, self.int_data, self.bool_data = self.prepare_data()

        # Define constants for column indices
        self.define_column_indices()

    def prepare_data(self):
        # Group columns by data type
        float_cols = []
        int_cols = []
        bool_cols = []
        
        for col in self.df.columns:
            dtype = self.df[col].dtype
            if dtype in [pl.Float32, pl.Float64]:
                float_cols.append(col)
            elif dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                           pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
                int_cols.append(col)
            elif dtype == pl.Boolean:
                bool_cols.append(col)
            else:
                raise ValueError(f"Unsupported data type: {dtype}")

        # Map column names to indices
        self.float_col_indices = {col: idx for idx, col in enumerate(float_cols)}
        self.int_col_indices = {col: idx for idx, col in enumerate(int_cols)}
        self.bool_col_indices = {col: idx for idx, col in enumerate(bool_cols)}

        # Convert columns to NumPy arrays
        float_data = np.column_stack([self.df[col].to_numpy(dtype=np.float64) for col in float_cols]) if float_cols else None
        int_data = np.column_stack([self.df[col].to_numpy(dtype=np.int64) for col in int_cols]) if int_cols else None
        bool_data = np.column_stack([self.df[col].to_numpy(dtype=np.bool_) for col in bool_cols]) if bool_cols else None

        return float_data, int_data, bool_data

    def define_column_indices(self):
        # Define indices as constants
        self.FLOAT_COL1_IDX = self.float_col_indices.get('float_col1', -1)
        self.INT_COL1_IDX = self.int_col_indices.get('int_col1', -1)
        self.BOOL_COL1_IDX = self.bool_col_indices.get('bool_col', -1)

    def run(self):
        # Run the outer loop and get the results
        total_scalar_outputs = outer_loop(
            self.float_data,
            self.int_data,
            self.bool_data,
            self.group_start,
            self.group_end,
            self.num_epochs,
            self.FLOAT_COL1_IDX,
            self.INT_COL1_IDX,
            self.BOOL_COL1_IDX
        )
        return total_scalar_outputs

# Numba-compiled functions

@njit
def expensive_function(float_data, int_data, bool_data, start_idx, end_idx,
                       scalar_output, FLOAT_COL1_IDX, INT_COL1_IDX, BOOL_COL1_IDX):
    """
    The inner loop function that performs computations on the data.
    """
    # Initialize scalar outputs
    scalar_output[0] = 0.0  # Sum of float_col1
    scalar_output[1] = 0.0  # Sum of int_col1
    scalar_output[2] = 0.0  # Sum of bool_col (True counts)

    for i in range(start_idx, end_idx):
        # Example computations using float data
        if float_data is not None and FLOAT_COL1_IDX != -1:
            scalar_output[0] += float_data[i, FLOAT_COL1_IDX]
            # Add more scalar computations as needed

        # Example computations using int data
        if int_data is not None and INT_COL1_IDX != -1:
            scalar_output[1] += int_data[i, INT_COL1_IDX]
            # Add more scalar computations as needed

        # Example computations using bool data
        if bool_data is not None and BOOL_COL1_IDX != -1:
            scalar_output[2] += bool_data[i, BOOL_COL1_IDX]
            # Add more scalar computations as needed

@njit
def outer_loop(float_data, int_data, bool_data, group_start, group_end, num_epochs,
               FLOAT_COL1_IDX, INT_COL1_IDX, BOOL_COL1_IDX):
    """
    The outer loop function that calls the inner loop multiple times.
    Stores scalar results of each epoch and group.
    """
    num_groups = len(group_start)
    n_results = 3  # Number of scalar outputs per group (adjust as needed)

    # Preallocate result array
    total_scalar_outputs = np.zeros((num_epochs, num_groups, n_results), dtype=np.float64)

    for epoch in range(num_epochs):
        for group_id in range(num_groups):
            start_idx = group_start[group_id]
            end_idx = group_end[group_id]

            scalar_output = np.zeros(n_results, dtype=np.float64)

            # Call the inner function
            expensive_function(
                float_data,
                int_data,
                bool_data,
                start_idx,
                end_idx,
                scalar_output,
                FLOAT_COL1_IDX,
                INT_COL1_IDX,
                BOOL_COL1_IDX
            )

            # Store scalar results
            total_scalar_outputs[epoch, group_id, :] = scalar_output

    return total_scalar_outputs

# Example usage

# Create an example Polars DataFrame with data
df = pl.DataFrame({
    "float_col1": [1.1, -2.2, 3.3, -4.4, 5.5, -6.6],
    "float_col2": [6.1, -7.2, 8.3, -9.4, 10.5, -11.6],
    "int_col1": [1, 2, 3, 4, 5, 6],
    "bool_col": [True, False, True, False, True, False],
    "uint_col": np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64),
})

# Group indices
group_start = [0, 3]  # Start indices of the groups
group_end = [3, 6]    # End indices of the groups

# Number of epochs (outer loop iterations)
num_epochs = 10  # Adjust as needed

# Instantiate the DataProcessor class
processor = DataProcessor(df, group_start, group_end, num_epochs)

# Run the processor to get the results
total_scalar_outputs = processor.run()

# Access and print results
for epoch in range(num_epochs):
    for group_id in range(len(group_start)):
        print(f"Epoch {epoch}, Group {group_id}, Scalar Output:", total_scalar_outputs[epoch, group_id])
