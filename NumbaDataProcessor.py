import numpy as np
import polars as pl
from numba import njit

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
        # Since Numba doesn't support dictionaries, we convert indices to arrays
        self.float_indices = np.array([self.float_col_indices[col] for col in sorted(self.float_col_indices)], dtype=np.int32)
        self.int_indices = np.array([self.int_col_indices[col] for col in sorted(self.int_col_indices)], dtype=np.int32)
        self.bool_indices = np.array([self.bool_col_indices[col] for col in sorted(self.bool_col_indices)], dtype=np.int32)

    def run(self):
        # Run the outer loop and get the results
        total_bool_outputs, total_scalar_outputs = outer_loop(
            self.float_data,
            self.int_data,
            self.bool_data,
            self.group_start,
            self.group_end,
            self.num_epochs,
            self.float_indices,
            self.int_indices,
            self.bool_indices
        )
        return total_bool_outputs, total_scalar_outputs

# Numba-compiled functions

@njit
def expensive_function(float_data, int_data, bool_data, start_idx, end_idx,
                       bool_output, scalar_output, float_indices, int_indices, bool_indices):
    """
    The inner loop function that performs computations on the data.
    """
    # Constants for column indices (adjust as per your columns)
    # Since we cannot use dynamic features inside njit, we need to define these constants outside
    FLOAT_COL1_IDX = 0  # float_indices[0]
    FLOAT_COL2_IDX = 1  # float_indices[1] if you have more columns
    INT_COL1_IDX = 0    # int_indices[0]
    BOOL_COL1_IDX = 0   # bool_indices[0]

    for i in range(start_idx, end_idx):
        idx = i - start_idx

        # Initialize condition
        condition = True

        # Example computations using float data
        if float_data is not None:
            # Access columns by indices
            # For example, float_col1 > 0
            condition = condition and (float_data[i, FLOAT_COL1_IDX] > 0)
            # Add more conditions or computations as needed

        # Example computations using int data
        if int_data is not None:
            condition = condition and (int_data[i, INT_COL1_IDX] % 2 == 0)
            # Add more conditions or computations as needed

        # Example computations using bool data
        if bool_data is not None:
            condition = condition and bool_data[i, BOOL_COL1_IDX]
            # Add more conditions or computations as needed

        # Update boolean output
        bool_output[idx] = condition

        # Update scalar outputs
        if float_data is not None:
            scalar_output[0] += float_data[i, FLOAT_COL1_IDX]
            # Add more scalar computations as needed
        if int_data is not None:
            scalar_output[1] += int_data[i, INT_COL1_IDX]
            # Add more scalar computations as needed
        if bool_data is not None:
            scalar_output[2] += bool_data[i, BOOL_COL1_IDX]
            # Add more scalar computations as needed

@njit
def outer_loop(float_data, int_data, bool_data, group_start, group_end, num_epochs,
               float_indices, int_indices, bool_indices):
    """
    The outer loop function that calls the inner loop multiple times.
    Stores results of each epoch in preallocated arrays.
    """
    num_groups = len(group_start)
    n_results = 5  # Number of scalar outputs per group (adjust as needed)

    # Preallocate result arrays
    total_bool_outputs = np.empty((num_epochs, num_groups), dtype=object)  # Each element is an array
    total_scalar_outputs = np.zeros((num_epochs, num_groups, n_results), dtype=np.float64)

    for epoch in range(num_epochs):
        for group_id in range(num_groups):
            start_idx = group_start[group_id]
            end_idx = group_end[group_id]
            n = end_idx - start_idx

            bool_output = np.empty(n, dtype=np.bool_)
            scalar_output = np.zeros(n_results, dtype=np.float64)

            # Call the inner function
            expensive_function(
                float_data,
                int_data,
                bool_data,
                start_idx,
                end_idx,
                bool_output,
                scalar_output,
                float_indices,
                int_indices,
                bool_indices
            )

            # Store results
            total_bool_outputs[epoch, group_id] = bool_output
            total_scalar_outputs[epoch, group_id, :] = scalar_output

    return total_bool_outputs, total_scalar_outputs

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
total_bool_outputs, total_scalar_outputs = processor.run()

# Access and print results
for epoch in range(num_epochs):
    for group_id in range(len(group_start)):
        print(f"Epoch {epoch}, Group {group_id}, Boolean Output:", total_bool_outputs[epoch, group_id])
        print(f"Epoch {epoch}, Group {group_id}, Scalar Output:", total_scalar_outputs[epoch, group_id])
