def find_strongly_connected_components(
    nodes: list[str], edges: list[dict[str, str]]
) -> list[list[str]]:
    # Build adjacency list and reversed adjacency list
    graph = {node: [] for node in nodes}
    reversed_graph = {node: [] for node in nodes}

    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        graph[src].append(tgt)
        reversed_graph[tgt].append(src)

    # Find SCCs using Kosaraju's algorithm
    visited = set()
    stack = []

    # First DFS to fill the stack
    def fill_order(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                fill_order(neighbor)
        stack.append(node)

    for node in nodes:
        if node not in visited:
            fill_order(node)

    # Second DFS to find SCCs
    visited.clear()  # Reset visited set for the second pass
    sccs = []

    def collect_scc(node, component):
        visited.add(node)
        component.append(node)
        for neighbor in reversed_graph[node]:
            if neighbor not in visited:
                collect_scc(neighbor, component)

    while stack:
        node = stack.pop()
        if node not in visited:
            component = []
            collect_scc(node, component)
            sccs.append(component)

    return sccs
