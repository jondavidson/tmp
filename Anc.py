def find_all_ancestors(dep_dict):
    # Dictionary to store all ancestors for each ID
    all_ancestors = {}

    def dfs(current_id, visited):
        # If the current ID is already computed, return its ancestors
        if current_id in all_ancestors:
            return all_ancestors[current_id]

        # If current ID has no ancestors, return an empty set
        if current_id not in dep_dict or not dep_dict[current_id]:
            all_ancestors[current_id] = set()
            return all_ancestors[current_id]

        # To avoid cycles, check if already visited
        if current_id in visited:
            raise ValueError(f"Cycle detected in the dependency graph at {current_id}!")

        # Mark as visited
        visited.add(current_id)

        # Collect ancestors recursively
        ancestors = set(dep_dict[current_id])  # Add immediate ancestors
        for ancestor in dep_dict[current_id]:
            ancestors.update(dfs(ancestor, visited))

        # Store the result and unmark visited
        all_ancestors[current_id] = ancestors
        visited.remove(current_id)
        return ancestors

    # Compute ancestors for all IDs
    for id_ in dep_dict:
        dfs(id_, set())

    return all_ancestors

# Example usage
dep_dict = {
    "A": ["B", "C"],
    "B": ["D"],
    "C": ["D", "E"],
    "D": ["F"],
    "E": [],
    "F": []
}

all_ancestors = find_all_ancestors(dep_dict)
print(all_ancestors)
