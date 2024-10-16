import numpy as np
import polars as pl
from numba import njit

def generate_dtype(df):
    """
    Generate a structured array dtype based on the DataFrame's schema.
    """
    dtype_list = []
    for col in df.columns:
        polars_dtype = df[col].dtype
        # Map Polars dtype to NumPy dtype
        if polars_dtype in [pl.Float32, pl.Float64]:
            np_dtype = np.float64
        elif polars_dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
            np_dtype = np.int64
        elif polars_dtype in [pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
            np_dtype = np.uint64
        elif polars_dtype == pl.Boolean:
            np_dtype = np.bool_
        else:
            raise ValueError(f"Unsupported data type: {polars_dtype} for column {col}")
        dtype_list.append((col, np_dtype))
    dtype = np.dtype(dtype_list)
    return dtype

# Create an example Polars DataFrame with data (you can replace this with your actual DataFrame)
df = pl.DataFrame({
    "float_col1": [1.1, -2.2, 3.3, -4.4, 5.5, -6.6],
    "float_col2": [6.1, -7.2, 8.3, -9.4, 10.5, -11.6],
    "int_col1": [1, 2, 3, 4, 5, 6],
    "bool_col": [True, False, True, False, True, False],
    "uint_col": np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64),
    # You can add or remove columns to change the schema
})

# Generate the dtype dynamically
dtype = generate_dtype(df)

# Now that the dtype is known, define Numba functions

@njit
def expensive_function(data, start_idx, end_idx, scalar_output):
    """
    The inner loop function that performs computations on the structured data.
    """
    for i in range(start_idx, end_idx):
        # Initialize or reset any condition or variables per iteration
        condition = True

        # Example computations using structured data
        # Loop over the fields in the dtype
        idx = 0  # Index for scalar_output
        for name in data.dtype.names:
            value = data[name][i]
            # Example: Sum values in scalar_output
            scalar_output[idx] += value

            # Example condition (modify as needed)
            if data.dtype[name] == np.float64:
                condition = condition and (value > 0)
            elif data.dtype[name] == np.int64:
                condition = condition and (value % 2 == 0)
            elif data.dtype[name] == np.uint64:
                condition = condition and (value > 2)
            elif data.dtype[name] == np.bool_:
                condition = condition and value
            # Add more conditions or computations as needed

            idx += 1  # Increment scalar_output index

        # If you need to store the condition result, you can do so (e.g., increment a counter)

@njit
def outer_loop(data, group_start, group_end, num_epochs):
    """
    The outer loop function that calls the inner loop multiple times.
    Stores scalar results of each epoch and group.
    """
    num_groups = len(group_start)
    n_results = len(data.dtype.names)  # Number of scalar outputs per group

    # Preallocate result array
    total_scalar_outputs = np.zeros((num_epochs, num_groups, n_results), dtype=np.float64)

    for epoch in range(num_epochs):
        for group_id in range(num_groups):
            start_idx = group_start[group_id]
            end_idx = group_end[group_id]

            scalar_output = np.zeros(n_results, dtype=np.float64)

            # Call the inner function
            expensive_function(
                data,
                start_idx,
                end_idx,
                scalar_output
            )

            # Store scalar results
            total_scalar_outputs[epoch, group_id, :] = scalar_output

    return total_scalar_outputs

# DataProcessor class to organize the data preparation and processing
class DataProcessor:
    def __init__(self, df, group_start, group_end, num_epochs):
        self.df = df
        self.group_start = group_start
        self.group_end = group_end
        self.num_epochs = num_epochs

        # Prepare structured data array
        self.structured_data = self.prepare_structured_data()

    def prepare_structured_data(self):
        """
        Convert the Polars DataFrame into a structured NumPy array
        with fields based on the generated dtype.
        """
        # Initialize a structured array with the same number of rows as the DataFrame
        data = np.zeros(len(self.df), dtype=dtype)

        # Populate the structured array with data from the DataFrame
        for name in dtype.names:
            data[name] = self.df[name].to_numpy()

        return data

    def run(self):
        # Run the outer loop and get the results
        total_scalar_outputs = outer_loop(
            self.structured_data,
            self.group_start,
            self.group_end,
            self.num_epochs
        )
        return total_scalar_outputs

# Example usage

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
