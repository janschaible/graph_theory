import unittest

from graphs.digraph import DiGraph

def get_simple_graph()-> DiGraph[str]:
    return DiGraph(
            ("a", "b"),
            ("a", "c"),
            ("b", "a")
        )

class DigraphTest(unittest.TestCase):
    def assert_graph_equal[T](self, graph: DiGraph[T], expected: dict[T, list[T]]):
        adjacency_list = graph.get_adjacency_list()
        assert adjacency_list == expected, f"expected: {expected}, \ngot:{adjacency_list}"


    def test_construction(self):
        graph = get_simple_graph()
        self.assert_graph_equal(
            graph,
            {
                "a" : ["b", "c"],
                "b" : ["a"],
                "c": []
            }
        )

    def test_delete_edge(self):
        graph = get_simple_graph()
        graph.delete_edge("a", "c")
        graph.delete_edge("b", "a")
        self.assert_graph_equal(
            graph,
            {
                "a": ["b"],
                "b": [],
                "c": []
            }
        )

    def test_delete_vertex(self):
        graph = get_simple_graph()
        graph.delete_vertex("b")
        self.assert_graph_equal(
            graph,
            {
                "a" : ["c"],
                "c": []
            }
        )

    def test_double_edge_premitted(self):
        graph = get_simple_graph()
        with self.assertRaises(AssertionError):
            graph.add_edge("a", "b")