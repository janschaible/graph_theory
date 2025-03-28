import unittest

from graphs.graph import Graph

def get_simple_graph()-> Graph[str]:
    return Graph(
            ("a", "b"),
            ("a", "c")
        )

def get_cyclic_graph(node_count: int) -> Graph[str]:
    assert node_count <= 24, "currently this implementation uses the alphabet"
    edges = [(chr(i), chr(i+1)) for i in range(ord("a"), ord("a") + node_count - 1)]
    edges.append((edges[0][0], edges[-1][-1]))
    return Graph(*edges)

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
        graph = get_cyclic_graph(3)
        assert graph.r_regular() == 2

    def test_is_path(self):
        graph = get_cyclic_graph(5)
        assert graph.is_path("a", "b", "c")
        assert not graph.is_path("a", "b", "d")
        assert not Graph(("a", "b"), ("b", "c"), ("c", "a"), ("a", "d")).is_path("a", "b", "d")

    def test_is_cycle(self):
        graph = get_cyclic_graph(3)
        assert graph.is_cycle("a", "b", "c")
        assert not graph.is_cycle("a", "b")
