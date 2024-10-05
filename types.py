from collections import OrderedDict

# Extended Polars to Numba type mapping
polars_to_numba = {
    "Int64": "int64",
    "Int32": "int32",
    "Float64": "float64",
    "Float32": "float32",
    "Utf8": "str",       # Numba uses 'str' for strings
    "Boolean": "bool"
}

def cast_polars_to_numba(polars_schema: OrderedDict, numba_spec: list):
    casted_schema = OrderedDict()

    # Iterate through each field in the Numba spec
    for field_name, numba_type in numba_spec:
        # Get the corresponding Polars type from the Polars schema
        polars_type = polars_schema.get(field_name)
        
        if polars_type is None:
            raise ValueError(f"Field {field_name} not found in Polars schema")
        
        # Check if the Numba type is a vector (has "[:]")
        if "[:]" in numba_type:
            # Extract the base type (e.g., float64 from float64[:])
            base_numba_type = numba_type.replace("[:]", "")
            target_polars_type = polars_to_numba.get(polars_type)

            if target_polars_type is None:
                raise ValueError(f"Unsupported Polars type: {polars_type} for field {field_name}")

            # Check if the base type matches after casting
            if target_polars_type != base_numba_type:
                print(f"Casting field '{field_name}' from {polars_type} to {base_numba_type}[]")
                casted_schema[field_name] = base_numba_type + "[]"
            else:
                casted_schema[field_name] = target_polars_type + "[]"
        else:
            # Scalar case
            target_numba_type = polars_to_numba.get(polars_type)
            
            if target_numba_type is None:
                raise ValueError(f"Unsupported Polars type: {polars_type} for field {field_name}")
            
            if target_numba_type != numba_type:
                print(f"Casting field '{field_name}' from {polars_type} to {numba_type}")
                casted_schema[field_name] = numba_type
            else:
                casted_schema[field_name] = target_numba_type

    return casted_schema

# Example usage:
polars_schema = OrderedDict([
    ('field1', 'Int64'),
    ('field2', 'Float32'),
    ('field3', 'Utf8'),
    ('field4', 'Boolean')
])

numba_spec = [
    ('field1', 'int64[:]'),
    ('field2', 'float64[:]'),
    ('field3', 'str'),
    ('field4', 'bool')
]

# Perform the cast
casted_schema = cast_polars_to_numba(polars_schema, numba_spec)
print(casted_schema)
