from typing import List, Any
from typing import Dict

import pandas as pd
from collections import deque


class PathFinder:
    def __init__(self, graph: Dict[str, List[str]]):
        self.graph = graph

    def find_shortest_path(self, start: str, end: str) -> list[str]:
        # Check for existence of start and end nodes in the graph
        if start not in self.graph or end not in self.graph:
            return []

        queue = deque([[start]])
        visited = {start}

        while queue:
            path = queue.popleft()
            node = path[-1]

            # Return path if we reach the end node
            if node == end:
                return path

            for neighbor in self.graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])

        return []  # No path found

    def are_connected(self, start: str, end: str) -> bool:
        path = self.find_shortest_path(start, end)
        return len(path) > 0
