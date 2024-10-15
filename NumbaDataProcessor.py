import numpy as np
import polars as pl
from numba import njit

# Constants for column indices (modify as per your DataFrame columns)
# For demonstration, we'll assume 5 float columns, 5 int columns, and 5 bool columns
FLOAT_COLS = {
    'float_col1': 0,
    'float_col2': 1,
    'float_col3': 2,
    'float_col4': 3,
    'float_col5': 4,
    # Add more as needed
}

INT_COLS = {
    'int_col1': 0,
    'int_col2': 1,
    'int_col3': 2,
    'int_col4': 3,
    'int_col5': 4,
    # Add more as needed
}

BOOL_COLS = {
    'bool_col1': 0,
    'bool_col2': 1,
    'bool_col3': 2,
    'bool_col4': 3,
    'bool_col5': 4,
    # Add more as needed
}

class NumbaDataProcessor:
    def __init__(self, df, group_start, group_end):
        """
        Initialize the DataProcessor with the DataFrame and group indices.
        """
        self.df = df
        self.group_start = group_start
        self.group_end = group_end
        # Prepare data
        self.float_cols, self.int_cols, self.bool_cols = self.group_columns_by_dtype()
        self.float_data, self.int_data, self.bool_data = self.convert_df_to_arrays()
    
    def group_columns_by_dtype(self):
        """
        Group DataFrame columns by their data types.
        """
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
                raise ValueError(f"Unsupported data type: {dtype} in column {col}")
        
        return float_cols, int_cols, bool_cols
    
    def convert_df_to_arrays(self):
        """
        Convert DataFrame columns to NumPy arrays.
        """
        float_data = np.column_stack([
            self.df[col].to_numpy(dtype=np.float64) for col in self.float_cols
        ]) if self.float_cols else None
        
        int_data = np.column_stack([
            self.df[col].to_numpy(dtype=np.int64) for col in self.int_cols
        ]) if self.int_cols else None
        
        bool_data = np.column_stack([
            self.df[col].to_numpy(dtype=np.bool_) for col in self.bool_cols
        ]) if self.bool_cols else None
        
        return float_data, int_data, bool_data
    
    def run(self, num_epochs):
        """
        Run the outer loop and return the accumulated results.
        """
        total_bool_outputs, total_scalar_outputs = outer_loop(
            self.float_data, self.int_data, self.bool_data,
            self.group_start, self.group_end, num_epochs
        )
        return total_bool_outputs, total_scalar_outputs

# Numba-compiled functions
@njit
def expensive_function(float_data, int_data, bool_data, start_idx, end_idx, bool_output, scalar_output):
    """
    The inner loop function that performs computations on the data.
    """
    # Column indices (using constants)
    FLOAT_COL1_IDX = FLOAT_COLS['float_col1']
    INT_COL1_IDX = INT_COLS['int_col1']
    BOOL_COL1_IDX = BOOL_COLS['bool_col1']
    # Add more indices as needed

    for row_idx in range(start_idx, end_idx):
        output_idx = row_idx - start_idx
        
        # Compute condition using multiple columns
        condition = True
        # Float conditions
        if float_data is not None:
            condition = condition and (float_data[row_idx, FLOAT_COL1_IDX] > 0)
            # Add more float conditions as needed
        # Int conditions
        if int_data is not None:
            condition = condition and (int_data[row_idx, INT_COL1_IDX] % 2 == 0)
            # Add more int conditions as needed
        # Bool conditions
        if bool_data is not None:
            condition = condition and bool_data[row_idx, BOOL_COL1_IDX]
            # Add more bool conditions as needed
        
        # Update outputs
        bool_output[output_idx] = condition
        
        # Scalar computations
        if float_data is not None:
            scalar_output[0] += float_data[row_idx, FLOAT_COL1_IDX]
            # Add more scalar computations as needed
        if int_data is not None:
            scalar_output[1] += int_data[row_idx, INT_COL1_IDX]
            # Add more scalar computations as needed
        if bool_data is not None:
            scalar_output[2] += bool_data[row_idx, BOOL_COL1_IDX]
            # Add more scalar computations as needed

@njit
def outer_loop(float_data, int_data, bool_data, group_start, group_end, num_epochs):
    """
    The outer loop function that calls the inner loop multiple times.
    Stores results of each epoch in preallocated arrays.
    """
    num_groups = len(group_start)
    n_results = 5  # Number of scalar outputs per group (adjust as needed)
    # Determine maximum group size for preallocation
    max_group_size = max(end - start for start, end in zip(group_start, group_end))
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
            
            expensive_function(float_data, int_data, bool_data, start_idx, end_idx, bool_output, scalar_output)
            
            # Store results
            total_bool_outputs[epoch, group_id] = bool_output
            total_scalar_outputs[epoch, group_id, :] = scalar_output
    
    return total_bool_outputs, total_scalar_outputs

# Example usage
if __name__ == '__main__':
    # Create an example DataFrame with 20 columns
    df = pl.DataFrame({
        'float_col1': np.random.randn(1000),
        'float_col2': np.random.randn(1000),
        'float_col3': np.random.randn(1000),
        'float_col4': np.random.randn(1000),
        'float_col5': np.random.randn(1000),
        'int_col1': np.random.randint(0, 100, size=1000),
        'int_col2': np.random.randint(0, 100, size=1000),
        'int_col3': np.random.randint(0, 100, size=1000),
        'int_col4': np.random.randint(0, 100, size=1000),
        'int_col5': np.random.randint(0, 100, size=1000),
        'bool_col1': np.random.choice([True, False], size=1000),
        'bool_col2': np.random.choice([True, False], size=1000),
        'bool_col3': np.random.choice([True, False], size=1000),
        'bool_col4': np.random.choice([True, False], size=1000),
        'bool_col5': np.random.choice([True, False], size=1000),
        # Add more columns as needed
    })

    # Define group indices (for example, divide data into 10 groups)
    num_groups = 10
    group_size = len(df) // num_groups
    group_start = [i * group_size for i in range(num_groups)]
    group_end = [(i + 1) * group_size for i in range(num_groups)]
    # Adjust last group_end if data size is not perfectly divisible
    group_end[-1] = len(df)
    
    # Initialize DataProcessor
    processor = DataProcessor(df, group_start, group_end)
    
    # Run the processing with the desired number of epochs
    num_epochs = 100  # Adjust as needed
    total_bool_outputs, total_scalar_outputs = processor.run(num_epochs)
    
    # Accessing results (example)
    # Print results from the first epoch and first group
    epoch = 0
    group_id = 0
    print(f"Epoch {epoch}, Group {group_id}, Boolean Output:", total_bool_outputs[epoch, group_id])
    print(f"Epoch {epoch}, Group {group_id}, Scalar Output:", total_scalar_outputs[epoch, group_id])
