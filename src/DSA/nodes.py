from typing import Any, NewType
import networkx as nx


# derived from https://github.com/langflow-ai/langflow/pull/5261
def find_last_node(nodes, edges):
    """This function receives a flow and returns the last node."""
    return next((n for n in nodes if all(e["source"] != n["id"] for e in edges)), None)


JsonRef = NewType("JsonRef", str)


# derived from https://github.com/pydantic/pydantic/pull/9650
def _get_all_json_refs(item: Any) -> set[JsonRef]:
    """Get all the definitions references from a JSON schema."""
    refs: set[JsonRef] = set()
    if isinstance(item, dict):
        for key, value in item.items():
            if key == "$ref" and isinstance(value, str):
                # the isinstance check ensures that '$ref' isn't the name of a property, etc.
                refs.add(JsonRef(value))
            elif isinstance(value, dict):
                refs.update(_get_all_json_refs(value))
            elif isinstance(value, list):
                for item in value:
                    refs.update(_get_all_json_refs(item))
    elif isinstance(item, list):
        for item in item:
            refs.update(_get_all_json_refs(item))
    return refs


# derived from https://github.com/langflow-ai/langflow/pull/5262
def find_cycle_vertices(edges):
    # Create a directed graph from the edges
    graph = nx.DiGraph(edges)

    # Find all simple cycles in the graph
    cycles = list(nx.simple_cycles(graph))

    # Flatten the list of cycles and remove duplicates
    cycle_vertices = {vertex for cycle in cycles for vertex in cycle}

    return sorted(cycle_vertices)


# derived from https://github.com/langflow-ai/langflow/pull/5263
def sort_chat_inputs_first(self, vertices_layers: list[list[str]]) -> list[list[str]]:
    # First check if any chat inputs have dependencies
    for layer in vertices_layers:
        for vertex_id in layer:
            if "ChatInput" in vertex_id and self.get_predecessors(
                self.get_vertex(vertex_id)
            ):
                return vertices_layers

    # If no chat inputs have dependencies, move them to first layer
    chat_inputs_first = []
    for layer in vertices_layers:
        layer_chat_inputs_first = [
            vertex_id for vertex_id in layer if "ChatInput" in vertex_id
        ]
        chat_inputs_first.extend(layer_chat_inputs_first)
        for vertex_id in layer_chat_inputs_first:
            # Remove the ChatInput from the layer
            layer.remove(vertex_id)

    if not chat_inputs_first:
        return vertices_layers

    return [chat_inputs_first, *vertices_layers]


def fibonacci(n: int) -> int:
    def memoize(f):
        memo = {}

        def helper(x):
            if x not in memo:
                memo[x] = f(x)
            return memo[x]

        return helper

    @memoize
    def fib(n: int) -> int:
        if n <= 1:
            return n
        return fib(n - 1) + fib(n - 2)

    return fib(n)
