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

    def test_num_of_vertices_with_odd_degree(self):
        graph = get_simple_graph()
        assert graph.number_of_vertices_with_odd_degree() == 2, "a and c should be of odd degree"

        graph = Graph(
            ("a", "b"),
            ("b", "c"),
            ("c", "d")
        )
        assert graph.number_of_vertices_with_odd_degree() == 2, "a and d should be of odd degree"

    def test_r_regular(self):
        graph = get_simple_graph()
        r = graph.r_regular()
        assert r is None, f"{graph.get_adjacency_list()} should not be regular but r was {r}"
        graph = Graph(
            ("a", "b"),
            ("b", "c"),
            ("c", "a"),
        )
        assert graph.r_regular() == 2
