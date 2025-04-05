from typing import override
from tests.graph_test import AbstractGraphTest
from graphs.graph import Graph
from graphs.linegraph import get_line_graph, LineGraphVertex
from graphs.graph_creators import get_k_graph

class TestLineGraph(AbstractGraphTest):
    @classmethod
    def get_render_dir(cls) -> str:
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

