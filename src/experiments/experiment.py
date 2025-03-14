# Function to find the node with highest degree (most connections)
def find_node_with_highest_degree(
    nodes: list[str], connections: dict[str, list[str]]
) -> str:
    # Initialize degrees with outgoing connections
    degree_count = {node: len(connections.get(node, [])) for node in nodes}

    # Update degrees for incoming connections
    for targets in connections.values():
        for target in targets:
            if target in degree_count:
                degree_count[target] += 1

    # Find the node with the maximum degree
    max_degree_node = max(degree_count, key=degree_count.get, default=None)

    return max_degree_node
