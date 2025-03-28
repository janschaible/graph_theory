from graphs.graph import Graph
from graphs.linegraph import get_line_graph, LineGraphVertex
import unittest
import pprint


def get_k_graph(node_count: int) -> Graph[str]:
    assert node_count <= 24, "currently this implementation uses the alphabet"
    vertices = [chr(i) for i in range(ord("a"), ord("a") + node_count)]

    edges: list[tuple[str, str]] = []
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            edges.append((vertices[i], vertices[j]))

    return Graph(*edges)

class GraphTest(unittest.TestCase):
    def assert_graph_equal[T](self, graph: Graph[T], expected: dict[T, list[T]]):
            adjacency_list = graph.get_adjacency_list()
    
            # Convert lists to sets so that order doesn't matter
            adjacency_set = {k: set(v) for k, v in adjacency_list.items()}
            expected_set = {k: set(v) for k, v in expected.items()}
            
            assert adjacency_set == expected_set, f"expected: {pprint.pformat(expected)}, \ngot:{pprint.pformat(adjacency_list)}"


    def test_construction(self):
        graph = get_k_graph(4)
        line_graph = get_line_graph(graph)
        self.assert_graph_equal(
            line_graph,
            {
                LineGraphVertex("a", "b"): [LineGraphVertex("a", "c"), LineGraphVertex("a", "d"), LineGraphVertex("b", "c"), LineGraphVertex("b", "d")],
                LineGraphVertex("a", "c"): [LineGraphVertex("a", "b"), LineGraphVertex("a", "d"), LineGraphVertex("b", "c"), LineGraphVertex("c", "d")],
                LineGraphVertex("a", "d"): [LineGraphVertex("a", "b"), LineGraphVertex("a", "c"), LineGraphVertex("b", "d"), LineGraphVertex("c", "d")],
                LineGraphVertex("b", "c"): [LineGraphVertex("a", "b"), LineGraphVertex("a", "c"), LineGraphVertex("b", "d"), LineGraphVertex("c", "d")],
                LineGraphVertex("b", "d"): [LineGraphVertex("a", "b"), LineGraphVertex("a", "d"), LineGraphVertex("b", "c"), LineGraphVertex("c", "d")],
                LineGraphVertex("c", "d"): [LineGraphVertex("a", "c"), LineGraphVertex("a", "d"), LineGraphVertex("b", "c"), LineGraphVertex("b", "d")]
            }
        )

