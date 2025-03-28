import unittest

from graphs.graph import Graph

def get_simple_graph()-> Graph[str]:
    return Graph(
            ("a", "b"),
            ("a", "c")
        )

class GraphTest(unittest.TestCase):
    def assert_graph_equal[T](self, graph: Graph[T], expected: dict[T, list[T]]):
        adjacency_list = graph.get_adjacency_list()
        assert adjacency_list == expected, f"expected: {expected}, \ngot:{adjacency_list}"


    def test_construction(self):
        graph = get_simple_graph()
        self.assert_graph_equal(
            graph,
            {
                "a": ["b", "c"],
                "b": ["a"],
                "c": ["a"]
            }
        )

    def test_delete_edge(self):
        graph = get_simple_graph()
        graph.delete_edge("a", "b")
        self.assert_graph_equal(
            graph,
            {
                "a": ["c"],
                "b": [],
                "c": ["a"]
            }
        )

    def test_delete_vertex(self):
        graph = get_simple_graph()
        graph.delete_vertex("b")
        self.assert_graph_equal(
            graph,
            {
                "a" : ["c"],
                "c": ["a"]
            }
        )
