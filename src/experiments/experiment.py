def graph_traversal(graph: dict[int, dict[int]], node: int) -> dict[int]:
    visited = set()  # Using a set for faster membership check
    result = []  # To maintain the order of traversal

    def dfs(n: int) -> None:
        if n in visited:
            return
        visited.add(n)
        result.append(n)  # Append to result when visited for the first time
        for neighbor in graph.get(n, []):
            dfs(neighbor)

    dfs(node)
    return result
