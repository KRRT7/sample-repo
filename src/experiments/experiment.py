from collections import defaultdict, deque


def calculate_node_betweenness(
    nodes: list[str], edges: list[dict[str, str]]
) -> dict[str, float]:
    betweenness = {node: 0.0 for node in nodes}
    adjacency = construct_adjacency(nodes, edges)

    for source in nodes:
        for target in nodes:
            if source == target:
                continue

            all_paths = bfs_shortest_paths(source, target, adjacency)

            for path in all_paths:
                for node in path[1:-1]:  # Exclude source and target
                    betweenness[node] += 1.0 / len(all_paths)

    return betweenness


def construct_adjacency(nodes, edges):
    adjacency = defaultdict(list)
    for edge in edges:
        adjacency[edge["source"]].append(edge["target"])
    return adjacency


def bfs_shortest_paths(source, target, adjacency):
    if source == target:
        return []

    all_paths = []
    queue = deque([(source, [source])])
    shortest_length = float("inf")

    while queue:
        current, path = queue.popleft()

        if len(path) > shortest_length:
            continue

        for neighbor in adjacency[current]:
            if neighbor in path:
                continue

            new_path = path + [neighbor]
            if neighbor == target:
                all_paths.append(new_path)
                shortest_length = len(new_path) - 1

            if len(new_path) <= shortest_length:
                queue.append((neighbor, new_path))

    return all_paths
