from numba import types

class DynamicJitClassFactory:
    def __init__(self, runtime_spec, class_name='DynamicJITClass'):
        """
        Initialize the factory with a runtime specification for class attributes.

        :param runtime_spec: Dictionary mapping attribute names to their types (as strings).
        :param class_name: Optional class name for the dynamically created class.
        """
        self.runtime_spec = runtime_spec
        self.class_name = class_name
        
        # Expanded type mapping to support a wider range of types
        self.type_mapping = {
            'int8': types.int8,
            'int16': types.int16,
            'int32': types.int32,
            'int64': types.int64,
            'uint8': types.uint8,
            'uint16': types.uint16,
            'uint32': types.uint32,
            'uint64': types.uint64,
            'float32': types.float32,
            'float64': types.float64,
            'complex64': types.complex64,
            'complex128': types.complex128,
            'boolean': types.boolean,
            'string': types.string,  # Numba's native string support
            'array_float64_1d': types.float64[:],  # Example for a 1D array of float64
            'array_int32_1d': types.int32[:],  # Example for a 1D array of int32
            'array_float32_2d': types.float32[:, :],  # Example for a 2D array of float32
            'array_int64_2d': types.int64[:, :],  # Example for a 2D array of int64
            # Add more types as necessary based on the use case
        }

        # Generate the jitclass specification
        self.jit_spec = self._create_jit_spec()

    def _create_jit_spec(self):
        """
        Internal method to generate a JIT class specification from the runtime spec.
        """
        return [(key, self.type_mapping[val]) for key, val in self.runtime_spec.items()]

    def create_class(self, methods=None):
        """
        Create a dynamically JIT-compiled class with the provided specification.

        :param methods: Optional dictionary of methods to add to the class.
        :return: The dynamically created JIT class.
        """
        # Dynamically construct the class body
        class_dict = {}

        # Add an initializer that sets the class attributes based on the runtime spec
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        class_dict['__init__'] = __init__

        # Add user-provided methods, if any
        if methods:
            class_dict.update(methods)

        # Dynamically create the class
        DynamicClass = type(self.class_name, (object,), class_dict)

        # Apply the JIT decorator to the class
        return jitclass(self.jit_spec)(DynamicClass)

# Example Usage:
if __name__ == "__main__":
    # Example runtime spec with different types
    runtime_spec = {
        'x': 'int32',
        'y': 'float64',
        'z': 'array_float64_1d'
    }

    # Optional methods to add to the class
    def sum(self):
        return self.x + self.y

    methods = {'sum': sum}

    # Create a factory instance
    factory = DynamicJitClassFactory(runtime_spec)

    # Create the dynamic JIT-compiled class
    DynamicJITClass = factory.create_class(methods)

    # Instantiate and use the class
    import numpy as np
    instance = DynamicJITClass(x=10, y=20.5, z=np.array([1.0, 2.0, 3.0]))
    print(instance.sum())  # Output will be 30.5
    print(instance.z)  # Accessing the 1D array attribute
