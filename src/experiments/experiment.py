# Function to find all leaf nodes (nodes with no outgoing edges)
def find_leaf_nodes(nodes: list[dict], edges: list[dict]) -> list[dict]:
    # Create a set of nodes with outgoing edges
    nodes_with_outgoing_edges = {edge["source"] for edge in edges}

    # Nodes that are not in the set of nodes with outgoing edges are leaf nodes
    leaf_nodes = [node for node in nodes if node["id"] not in nodes_with_outgoing_edges]

    return leaf_nodes
