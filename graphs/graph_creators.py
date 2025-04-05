from graphs.graph import Graph

def get_cyclic_graph(node_count: int) -> Graph[str]:
    assert node_count <= 24, "currently this implementation uses the alphabet"
    edges = [(chr(i), chr(i+1)) for i in range(ord("a"), ord("a") + node_count - 1)]
    edges.append((edges[0][0], edges[-1][-1]))
    return Graph(*edges)

def get_k_graph(node_count: int) -> Graph[str]:
    assert node_count <= 24, "currently this implementation uses the alphabet"
    vertices = [chr(i) for i in range(ord("a"), ord("a") + node_count)]

    edges: list[tuple[str, str]] = []
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            edges.append((vertices[i], vertices[j]))

    return Graph(*edges)

