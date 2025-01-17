import numpy as np
from numba import njit

@njit
def construct_hierarchy_and_levels(orders):
    # Preprocessing the orders into NumPy arrays
    num_orders = len(orders)
    order_ids = np.zeros(num_orders, dtype=np.int32)
    parent_ids = np.zeros(num_orders, dtype=np.int32)

    for i in range(num_orders):
        order_ids[i] = orders[i][0]
        parent_ids[i] = orders[i][1]

    # Initialize results: levels and ancestors
    levels = -np.ones(num_orders, dtype=np.int32)  # Initialize levels with -1
    max_ancestors = 10  # Maximum depth of hierarchy (arbitrary, can be increased)
    ancestors = -np.ones((num_orders, max_ancestors), dtype=np.int32)

    # Recursive function to calculate hierarchy
    def find_hierarchy_and_levels(order_idx, current_level, ancestor_list):
        # Update level
        levels[order_idx] = current_level

        # Update ancestors
        for i in range(len(ancestor_list)):
            ancestors[order_idx, i] = ancestor_list[i]

        # Find children of the current order
        current_order_id = order_ids[order_idx]
        children_idx = np.where(parent_ids == current_order_id)[0]

        # Process each child
        for child_idx in children_idx:
            find_hierarchy_and_levels(child_idx, current_level + 1, ancestor_list + [current_order_id])

    # Identify root orders
    root_indices = np.where(parent_ids == -1)[0]
    for root_idx in root_indices:
        find_hierarchy_and_levels(root_idx, 0, [])

    return levels, ancestors

# Example usage
orders = np.array([
    [1, -1],  # orderId=1, parentId=-1 (root)
    [2, 1],   # orderId=2, parentId=1
    [3, 1],   # orderId=3, parentId=1
    [4, 2],   # orderId=4, parentId=2
    [5, 2],   # orderId=5, parentId=2
    [6, 3],   # orderId=6, parentId=3
], dtype=np.int32)

# Call the function
levels, ancestors = construct_hierarchy_and_levels(orders)

# Display the results
print("Levels:", levels)
print("Ancestors:")
print(ancestors)
