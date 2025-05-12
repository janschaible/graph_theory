import numpy as np
import networkx as nx
from graphs.tree import Tree
from tests.graph_test import AbstractGraphTest
from graphs.digraph import DiGraph


def get_simple_graph() -> DiGraph[str]:
    return DiGraph(("a", "b", 1), ("a", "c", 2), ("b", "a", 3))


class TestDigraph(AbstractGraphTest):
    @classmethod
    def get_render_dir(cls) -> str:
        return "test_digraph"

    def test_construction(self):
        graph = get_simple_graph()
        self.assert_graph_equal(graph, {"a": ["b", "c"], "b": ["a"], "c": []})
        self.render(graph, "test_construction")

    def test_delete_edge(self):
        graph = get_simple_graph()
        graph.delete_edge("a", "c")
        graph.delete_edge("b", "a")
        self.assert_graph_equal(graph, {"a": ["b"], "b": [], "c": []})
        self.assert_weight_not_present(graph, "a", "c")
        self.assert_weight_not_present(graph, "b", "a")
        assert graph.get_weight("a", "b") == 1

    def test_delete_vertex(self):
        graph = get_simple_graph()
        graph.delete_vertex("b")
        self.assert_graph_equal(graph, {"a": ["c"], "c": []})

    def test_double_edge_premitted(self):
        graph = get_simple_graph()
        with self.assertRaises(AssertionError):
            graph.add_edge("a", "b")

    def test_weights(self):
        graph = DiGraph(
            ("a", "b", 1),
            ("b", "c", 2),
            ("c", "d", 3),
        )
        assert graph.get_weight("a", "b") == 1
        assert graph.get_weight("b", "c") == 2
        assert graph.get_weight("c", "d") == 3
        self.render(graph, "test_weights")

    def test_adjacency_matrix(self):
        graph = DiGraph(
            ("a", "b"),
            ("b", "c"),
            ("c", "d"),
        )
        assert np.array_equal(
            graph.get_adjacency_matrix(),
            np.array(
                [
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                ]
            ),
        )


    def test_floyd_warshall(self):
        graph = DiGraph(
            (1, 2, 4),
            (1, 3, 2),
            (2, 4, 3),
            (3, 2, -1),
            (4, 5, 2),
            (5, 4, 1)
        )
        res,_,_,_ = graph.floyd_warshall()
        assert np.array_equal(
            res,
            np.array([
            [ 0 , 1  , 2 , 4 , 6 ],
            [ float("inf") , 0  , float("inf") , 3 , 5 ],
            [ float("inf") , -1 , 0 , 2 , 4 ],
            [ float("inf") , float("inf")  , float("inf") , 0 , 2 ],
            [ float("inf") , float("inf")  , float("inf") , 1 , 0 ],
            ])
        )

    def test_floyd_warshall_negative_cycle(self):
        graph = DiGraph(
            (1,2 , 3),
            (2,1 , 1.3),
            (2, 3, -1),
            (2,4 ,7 ),
            (5,2 , -1),
            (3,4 , -1),
            (4,5 , 1.2),
            (1, 5, 2),
        )
        self.render(graph, "test_floyd_warshall")
        res,_,_,_ = graph.floyd_warshall()
        print(res)
        assert np.array_equal(
            res,
            [4, 3, 2, 5, 4]
        )
    
    def test_betweeness(self):
        # todo js this shit is broken...
        graph = DiGraph(
            ("A", "B", 1),
            ("A", "E", 1),
            ("B", "G", 1),
            ("E", "G", 1),
            ("A", "H", 1),
            ("G", "C", 1),
            ("G", "F", 1),
            ("G", "K", 1),
            ("F", "D", 1),
            ("C", "D", 1),
            ("K", "L", 1),
            ("L", "D", 1),
            ("H", "I", 1),
            ("I", "J", 1),
            ("J", "D", 1),
        )
        self.render(graph, "test_betweeness")
        print(nx.betweenness_centrality(graph.to_network_x()))
        print(graph.betweeness_centrality())

    def test_bfs(self):
        graph = DiGraph(
            (1,2),
            (1,5),
            (2,1),
            (2,4),
            (2,5),
            (4,2),
            (4,6),
            (6,2),
            (6,5),
            (5,3),
            (5,7),
            (3,7),
            (7,5)
        )
        tree,girth = graph.bfs(1)
        assert tree == Tree(
            1,
            [
                Tree(2, [Tree(4, [Tree(6,[], "1246")], "124")], "12"),
                Tree(5, [Tree(3,[], "153"), Tree(7,[], "157")], "15")
            ],
            "1"
        )
        assert girth == 2

    def test_girth(self):
        graph = DiGraph(
            (1,2),
            (1,5),
            (2,4),
            (2,5),
            (4,6),
            (6,2),
            (6,5),
            (5,3),
            (3,7),
            (7,5)
        )
        _,girth = graph.bfs(1)
        self.render(graph, "test_girth")

        print(girth)
        assert girth == 3