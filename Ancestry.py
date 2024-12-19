import os
import numpy as np
import numba

# Example dataset: A small hierarchy of orders.
# Columns: OrderId, ParentId, RootId
records = np.array([
    (10, -1, 10),  # Root order (OrderId=10, no parent)
    (20, 10, 10),  # Child of 10
    (30, 20, 10),  # Child of 20
    (40, 20, 10),  # Another child of 20
    (50, 10, 10),  # Another child of 10
    (60, 50, 10),  # Child of 50
], dtype=[('OrderId', 'i8'), ('ParentId', 'i8'), ('RootId', 'i8')])

# Extract arrays
order_ids = records['OrderId']
parent_ids = records['ParentId']
n = len(order_ids)

# Create a mapping from OrderId to contiguous indices
sorted_indices = np.argsort(order_ids)
sorted_order_ids = order_ids[sorted_indices]
max_oid = sorted_order_ids[-1]
orderid_to_index = np.full(max_oid+1, -1, dtype=np.int64)
for i, oid in enumerate(sorted_order_ids):
    orderid_to_index[oid] = i

# Reindex parent_ids to compact indices
compact_parent_ids = np.empty(n, dtype=np.int64)
for i in range(n):
    pid = parent_ids[i]
    if pid == -1:
        compact_parent_ids[i] = -1
    else:
        compact_parent_ids[i] = orderid_to_index[pid]

# Build adjacency list (children arrays)
child_counts = np.zeros(n, dtype=np.int64)
for i in range(n):
    p = compact_parent_ids[i]
    if p != -1:
        child_counts[p] += 1

child_indices = np.empty(child_counts.sum(), dtype=np.int64)
child_starts = np.empty(n, dtype=np.int64)
child_ends = np.empty(n, dtype=np.int64)

current_pos = 0
for i in range(n):
    child_starts[i] = current_pos
    current_pos += child_counts[i]
    child_ends[i] = current_pos

child_counts.fill(0)
for i in range(n):
    p = compact_parent_ids[i]
    if p != -1:
        idx = p
        start = child_starts[idx]
        ccount = child_counts[idx]
        child_indices[start + ccount] = i
        child_counts[idx] += 1

@numba.njit
def find_ancestors(node_index: int, parent_ids: np.ndarray):
    """
    Return a list of ancestors for the given node_index.
    Ancestors are returned as a Python list of indices from parent up to root.
    """
    ancestors = []
    current = node_index
    while True:
        p = parent_ids[current]
        if p == -1:
            break
        ancestors.append(p)
        current = p
    return ancestors

@numba.njit
def find_descendants(node_index: int, child_indices: np.ndarray, child_starts: np.ndarray, child_ends: np.ndarray):
    """
    Return a list of all descendants of node_index using BFS.
    """
    queue = [node_index]
    descendants = []
    head = 0
    while head < len(queue):
        current = queue[head]
        head += 1
        start = child_starts[current]
        end = child_ends[current]
        for i in range(start, end):
            c = child_indices[i]
            descendants.append(c)
            queue.append(c)
    return descendants

if __name__ == "__main__":
    # Test the functions
    # Test ancestors of OrderId=30
    test_oid = 30
    test_index = orderid_to_index[test_oid]
    anc = find_ancestors(test_index, compact_parent_ids)
    anc_oids = order_ids[anc]
    print(f"Ancestors of {test_oid}:", anc_oids)

    # Test descendants of OrderId=10 (the root)
    root_oid = 10
    root_index = orderid_to_index[root_oid]
    desc = find_descendants(root_index, child_indices, child_starts, child_ends)
    desc_oids = order_ids[desc]
    print(f"Descendants of {root_oid}:", np.sort(desc_oids))

    # Test run with NUMBA_DISABLE_JIT=1:
    # In a terminal: NUMBA_DISABLE_JIT=1 python test_hierarchy.py
    # Should still produce correct results (though slower).
