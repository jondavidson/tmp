import polars as pl
import numpy as np
from numba import jitclass, float64, uint8, types, numpy_support

# Define the Numba jitclass spec
spec = [
    ('col1', float64[:]),
    ('col2', uint8[:]),
    # Add more columns and types as needed
]

# Extract parameter names from the spec
param_names = [col_name for col_name, _ in spec]

# Build the class definition code as a string
class_def = """
@jitclass(spec)
class MyClass:
    def __init__(self, {}):
{}
""".format(
    ', '.join(param_names),
    '\n'.join(['        self.{0} = {0}'.format(name) for name in param_names])
)

# Execute the class definition code
local_vars = {'spec': spec, 'jitclass': jitclass}
exec(class_def, globals(), local_vars)
MyClass = local_vars['MyClass']

# Assume you have a Polars DataFrame 'df'
df = pl.DataFrame({
    'col1': [1.1, 2.2, 3.3],
    'col2': [1, 0, 1],
    # Add more data as needed
})

# Function to map Numba types to NumPy dtypes
def numba_type_to_numpy_dtype(numba_type):
    if isinstance(numba_type, types.Array):
        numba_dtype = numba_type.dtype
    else:
        numba_dtype = numba_type
    numpy_dtype = numpy_support.as_dtype(numba_dtype)
    return numpy_dtype

# Extract and cast columns based on the spec
kwargs = {}
for col_name, numba_type in spec:
    numpy_dtype = numba_type_to_numpy_dtype(numba_type)
    try:
        kwargs[col_name] = df[col_name].to_numpy().astype(numpy_dtype)
    except ValueError as e:
        print(f"Error casting column '{col_name}': {e}")
        raise

# Now, construct your jitclass instance using unpacking
my_instance = MyClass(**kwargs)

