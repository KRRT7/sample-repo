from collections import defaultdict, deque


def find_node_clusters(nodes: list[dict], edges: list[dict]) -> list[list[dict]]:
    node_map = {node["id"]: node for node in nodes}

    # Create an adjacency list
    adjacency = defaultdict(list)
    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        adjacency[src].append(tgt)
        adjacency[tgt].append(src)

    # Track visited nodes
    visited = set()
    clusters = []

    for node in nodes:
        node_id = node["id"]
        if node_id in visited:
            continue

        # Start a new cluster
        cluster = []
        queue = deque([node_id])
        visited.add(node_id)

        while queue:
            current = queue.popleft()
            cluster.append(node_map[current])

            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        clusters.append(cluster)

    return clusters
