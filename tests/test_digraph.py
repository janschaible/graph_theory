import math
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
    
    def test_ford_and_fulkerson(self):
        graph = DiGraph(
            ("s","a",0,10),
            ("a","d",0,10),
            ("d","s",0,5),
            ("d","c",0,6),
            ("d","e",0,5),
            ("c","a",0,6),
            ("c","t",0,4),
            ("e","b",0,5),
            ("e","t",0,5),
            ("b","d",0,4),
            vertices=["s", "a", "d", "e", "c", "b", "t"]
        )
        f,S,T = graph.ford_and_fulkerson("s", "t")
        print(f)
        print(S)
        print(T)

    def test_zwick_ford_fulkerson(self):
        r = (math.sqrt(5)-1)/2
        graph = DiGraph(
            ("s","d",0,4),
            ("s","b",0,4),
            ("s","a",0,4),
            ("d","c",0,r),
            ("d","t",0,4),
            ("c","t",0,4),
            ("b","c",0,1),
            ("b","a",0,1),
            ("a","t",0,4),
            vertices=["s", "d", "c", "b", "a", "t"]
        )
        self.render(graph, "test_zwick_ford_fulkerson", render_capacities=True)
        f,S,T = graph.ford_and_fulkerson("s", "t")
        print(f)
        print(S)
        print(T)

    def test_auxnet(self):
        graph = DiGraph(
            (1,8,7,7),
            (1,6,0,3),
            (1,10,3,3),
            (3,8,0,9),
            (3,5,0,4),
            (5,2,7,8),
            (6,9,0,2),
            (6,8,0,2),
            (7,8,0,9),
            (8,5,7,7),
            (9,2,3,3),
            (9,8,0,9),
            (9,7,0,6),
            (10,9,3,6),
            (10,3,0,3),
            (10,5,0,3),
            vertices=[1,2,3,4,5,6,7,8,9,10]
        )
        _, edges, is_max, depth = graph.auxnet(1,2)
        print(edges)
        assert is_max == False
        assert depth == 6
