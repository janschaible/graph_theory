from typing import override
from tests.graph_test import AbstractGraphTest
from graphs.graph import Graph
from graphs.linegraph import get_line_graph, LineGraphVertex


def get_k_graph(node_count: int) -> Graph[str]:
    assert node_count <= 24, "currently this implementation uses the alphabet"
    vertices = [chr(i) for i in range(ord("a"), ord("a") + node_count)]

    edges: list[tuple[str, str]] = []
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            edges.append((vertices[i], vertices[j]))

    return Graph(*edges)

class TestLineGraph(AbstractGraphTest):
    @override
    def get_render_dir(self) -> str:
        return "test_linegraph"

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
        self.render(graph, "test_construction_original")
        self.render(line_graph, "test_construction_line_graph")

