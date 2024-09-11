import os
import importlib.util
from numba import jitclass, int32, float64

# Step 1: Create the .py file dynamically
def create_class_file(file_name, class_name, attributes):
    with open(file_name, 'w') as f:
        f.write("from numba import jitclass, int32, float64\n")
        f.write("spec = [\n")
        for attr, attr_type in attributes.items():
            f.write(f"    ('{attr}', {attr_type}),\n")
        f.write("]\n\n")
        f.write(f"@jitclass(spec)\n")
        f.write(f"class {class_name}:\n")
        f.write("    def __init__(self, " + ", ".join(attributes.keys()) + "):\n")
        for attr in attributes.keys():
            f.write(f"        self.{attr} = {attr}\n")
        f.write("\n")
        f.write("    def sum(self):\n")
        f.write("        return " + " + ".join([f"self.{attr}" for attr in attributes.keys()]) + "\n")

# Step 2: Create the class file dynamically
file_name = "dynamic_class.py"
class_name = "DynamicJITClass"
attributes = {
    'x': 'int32',
    'y': 'float64',
    'z': 'float64'
}

create_class_file(file_name, class_name, attributes)

# Step 3: Dynamically import the generated module
def dynamic_import(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import the generated dynamic_class.py module
dynamic_module = dynamic_import("dynamic_class", file_name)

# Step 4: Use the class from the dynamically created module
DynamicJITClass = getattr(dynamic_module, class_name)

# Instantiate and use the dynamically created and compiled class
instance = DynamicJITClass(1, 2.5, 3.0)
print(instance.sum())  # This should output the sum of 1 + 2.5 + 3.0 = 6.5

# Optional: Clean up the generated file
os.remove(file_name)
