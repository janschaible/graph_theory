from graphs.digraph import DiGraph
from graphs.graph_creators import get_k_graph
from graphs.algorithms.hierholzer import Hierholzer
from tests.graph_test import AbstractGraphTest

class TestHierholzer(AbstractGraphTest):
    @classmethod
    def get_render_dir(cls) -> str:
        return "tests/algorithms/test_graphs/test_hierholzer"

    def test_hierholzer(self):
        graph = get_k_graph(5)
        hierholzer = Hierholzer(graph, f"{self.get_render_dir()}/test_hierholzer")
        hierholzer.solve()


    def test_hierholzer_digraph(self):
        graph = DiGraph(
            ("a", "b"),
            ("b", "c"),
            ("c", "d"),
            ("d", "e"),
            ("e", "a"),
            ("a", "c"),
            ("c", "e"),
            ("e", "b"),
            ("b", "d"),
            ("d", "a"),
        )
        hierholzer = Hierholzer(graph, f"{self.get_render_dir()}/test_hierholzer_digraph")
        hierholzer.solve()
