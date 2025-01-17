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


import polars as pl

def construct_ancestors(df: pl.DataFrame, max_depth: int = 10) -> pl.DataFrame:
    # Initialize the ancestors columns with null values
    for depth in range(max_depth):
        df = df.with_column(pl.lit(None).alias(f'ancestor_{depth}'))

    # Iterate through levels to populate ancestors
    for current_level in range(1, df['level'].max() + 1):
        # Filter current level orders
        current_level_orders = df.filter(df['level'] == current_level)

        # Join to find parents and update ancestors
        df = df.join(
            current_level_orders.select(['orderId', 'parentId'] + [f'ancestor_{i}' for i in range(max_depth)]),
            left_on='parentId',
            right_on='orderId',
            how='left',
            suffix='_parent'
        ).with_columns([
            pl.when(pl.col(f'ancestor_{i}_parent').is_not_null())
            .then(pl.col(f'ancestor_{i}_parent'))
            .otherwise(pl.col(f'ancestor_{i}'))
            .alias(f'ancestor_{i}')
            for i in range(max_depth - 1)
        ])

        # Cleanup parent-related columns
        df = df.drop([f'ancestor_{i}_parent' for i in range(max_depth)] + ['orderId_parent', 'parentId_parent'])

    return df

# Example Usage
data = [
    {"orderId": 1, "parentId": -1, "rootId": 1, "level": 0},
    {"orderId": 2, "parentId": 1, "rootId": 1, "level": 1},
    {"orderId": 3, "parentId": 1, "rootId": 1, "level": 1},
    {"orderId": 4, "parentId": 2, "rootId": 1, "level": 2},
    {"orderId": 5, "parentId": 2, "rootId": 1, "level": 2},
    {"orderId": 6, "parentId": 3, "rootId": 1, "level": 2},
]

# Create a Polars DataFrame
df = pl.DataFrame(data)

# Construct ancestors
result = construct_ancestors(df)

# Display result






import numpy as np
from numba import njit

@njit
def construct_hierarchy_and_levels_iterative(orders, max_depth=10):
    # Extract order details
    num_orders = len(orders)
    order_ids = orders[:, 0]
    parent_ids = orders[:, 1]
    levels = orders[:, 3]

    # Initialize the ancestor array
    ancestors = -np.ones((num_orders, max_depth), dtype=np.int32)

    # Sort orders by level for sequential processing
    sorted_indices = np.argsort(levels)
    orders = orders[sorted_indices]

    # Map orderId to index for quick lookup
    order_id_to_index = {order_ids[i]: i for i in range(num_orders)}

    # Iterate over orders level by level
    for i in range(num_orders):
        order_id = orders[i, 0]
        parent_id = orders[i, 1]
        level = levels[i]

        if parent_id != -1:
            parent_index = order_id_to_index[parent_id]

            # Copy parent's ancestors and add parent to the list
            ancestors[i, :level] = ancestors[parent_index, :level]
            ancestors[i, level - 1] = parent_id

    # Reverse sorting to match original order
    reverse_indices = np.argsort(sorted_indices)
    ancestors = ancestors[reverse_indices]

    return ancestors

# Example Usage
orders = np.array([
    [1, -1, 1, 0],  # orderId=1, parentId=-1, rootId=1, level=0
    [2, 1, 1, 1],   # orderId=2, parentId=1, rootId=1, level=1
    [3, 1, 1, 1],   # orderId=3, parentId=1, rootId=1, level=1
    [4, 2, 1, 2],   # orderId=4, parentId=2, rootId=1, level=2
    [5, 2, 1, 2],   # orderId=5, parentId=2, rootId=1, level=2
    [6, 3, 1, 2],   # orderId=6, parentId=3, rootId=1, level=2
], dtype=np.int32)

# Call the function
ancestors = construct_hierarchy_and_levels_iterative(orders)

# Display the results
print("Ancestors:")
print(ancestors)
