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
    # Lists to accumulate chat inputs and vertices with dependencies
    chat_inputs_first = []
    layers_with_dependencies = []

    for layer in vertices_layers:
        layer_chat_inputs_first = []
        layer_with_dependencies = []

        for vertex_id in layer:
            if "ChatInput" in vertex_id:
                if self.get_predecessors(self.get_vertex(vertex_id)):
                    # If any ChatInput has dependencies, return the original layers
                    return vertices_layers
                layer_chat_inputs_first.append(vertex_id)
            else:
                layer_with_dependencies.append(vertex_id)

        chat_inputs_first.extend(layer_chat_inputs_first)
        layers_with_dependencies.append(layer_with_dependencies)

    if not chat_inputs_first:
        return vertices_layers

    # Filter out empty layers
    non_empty_layers = [layer for layer in layers_with_dependencies if layer]

    return [chat_inputs_first, *non_empty_layers]
