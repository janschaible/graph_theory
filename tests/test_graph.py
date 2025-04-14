from typing import override
from tests.graph_test import AbstractGraphTest
from graphs.graph import Graph
from graphs.graph_creators import get_cyclic_graph

def get_simple_graph()-> Graph[str]:
    return Graph(
            ("a", "b"),
            ("a", "c")
        )

class TestGraph(AbstractGraphTest):

    @classmethod
    def get_render_dir(cls) -> str:
        return "test_graph"

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
        self.render(graph, "test_construction")

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
        self.render(graph, "test_is_path")

    def test_is_cycle(self):
        graph = get_cyclic_graph(3)
        assert graph.is_cycle("a", "b", "c")
        assert not graph.is_cycle("a", "b")

    def test_is_euler_tour(self):
        graph = Graph(
            ("a", "b"),
            ("b", "c"),
            ("c", "a"),
            ("a", "d"),
        )
        assert graph.is_euler_tour("a", "b", "c", "a") == False
        graph.delete_vertex("d")
        assert graph.is_euler_tour("a", "b", "c", "a")
        assert graph.is_euler_tour("a", "b", "c", "b", "c" "a") == False

    def test_is_hamiltonian(self):

        graph = Graph(
            ("a", "b"),
            ("b", "c"),
            ("c", "a"),
            ("c", "d"),
            ("d", "a"),
        )
        assert graph.is_hamiltonean("a", "b", "c", "a") == False
        assert graph.is_hamiltonean("a", "b", "c", "d", "a")

    def test_weights(self):
        graph = Graph(
            ("a", "b", 1),
            ("b", "c", 2),
            ("c", "d", 3),
        )
        assert graph.get_weight("a", "b") == 1
        assert graph.get_weight("b", "a") == 1
        assert graph.get_weight("b", "c") == 2
        assert graph.get_weight("c", "b") == 2
        assert graph.get_weight("c", "d") == 3
        assert graph.get_weight("d", "c") == 3
        self.render(graph, "test_weights")

    def test_eigen_centrality(self):
        graph = Graph(
            (1,8),
            (1,7),
            (2,4),
            (2,6),
            (2,7),
            (4,6),
            (4,7),
            (4,8),
            (6,9),
            (9,3),
            (9,5),
            (8,5),
            (7,5),
            (5,3),
        )
        self.render(graph, "test_eigen_centrality", eigen_centrality=True)