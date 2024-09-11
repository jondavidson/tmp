import os
import importlib.util
from numba import jitclass, int32, float64, types

class DynamicJitClassFactory:
    def __init__(self, class_name="DynamicJITClass", file_name="dynamic_class.py"):
        """
        Initializes the DynamicJitClassFactory.

        :param class_name: The name of the dynamically created class.
        :param file_name: The file name for the dynamically created Python class.
        """
        self.class_name = class_name
        self.file_name = file_name

    def create_class_file(self, attributes):
        """
        Creates a Python file that defines a jitclass with the given attributes.

        :param attributes: A dictionary where keys are attribute names, and values are their numba types.
        """
        with open(self.file_name, 'w') as f:
            f.write("from numba import jitclass, int32, float64, types\n")
            f.write("spec = [\n")
            for attr, attr_type in attributes.items():
                f.write(f"    ('{attr}', {attr_type}),\n")
            f.write("]\n\n")
            f.write(f"@jitclass(spec)\n")
            f.write(f"class {self.class_name}:\n")
            f.write("    def __init__(self, " + ", ".join(attributes.keys()) + "):\n")
            for attr in attributes.keys():
                f.write(f"        self.{attr} = {attr}\n")
            f.write("\n")
            f.write("    def sum_first_row(self):\n")
            f.write("        return self.arr[0, :].sum()\n")  # Example for summing the first row of a 2D array

    def dynamic_import(self):
        """
        Dynamically imports the module containing the generated class.

        :return: The dynamically imported module.
        """
        spec = importlib.util.spec_from_file_location("dynamic_class", self.file_name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def instantiate_class(self, attributes, *args):
        """
        Creates the class file, imports the class, and instantiates it with the provided arguments.

        :param attributes: A dictionary of attribute names and their numba types.
        :param args: The arguments to initialize the dynamically created class.
        :return: An instance of the dynamically created class.
        """
        # Step 1: Create the class file dynamically
        self.create_class_file(attributes)

        # Step 2: Dynamically import the generated class
        dynamic_module = self.dynamic_import()

        # Step 3: Get the class and instantiate it
        DynamicJITClass = getattr(dynamic_module, self.class_name)
        instance = DynamicJITClass(*args)

        # Optional: Clean up by removing the dynamically created Python file
        self.cleanup()

        return instance

    def cleanup(self):
        """
        Cleans up the dynamically generated Python file.
        """
        if os.path.exists(self.file_name):
            os.remove(self.file_name)


# Example usage of DynamicJitClassFactory with a 2D array
if __name__ == "__main__":
    import numpy as np
    # Initialize the factory
    factory = DynamicJitClassFactory()

    # Define the attributes and their numba types (including a 2D float64 array)
    attributes = {
        'x': 'int32',
        'arr': 'float64[:,:]',  # 2D array of float64
    }

    # Create a 2D array to pass to the class
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])

    # Instantiate the dynamically created class
    instance = factory.instantiate_class(attributes, 1, arr)

    # Use the instance (calling the sum_first_row method to sum the first row of the 2D array)
    print(instance.sum_first_row())  # Output: 3.0 (sum of 1.0 and 2.0)
