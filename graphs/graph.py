from typing import TypeVar, override, Optional
from graphs.digraph import DiGraph, EdgeDefinition
from copy import deepcopy
import networkx as nx

T = TypeVar("T")

class Graph(DiGraph[T]):
    def __init__(self, *edges: EdgeDefinition, **kwargs) -> None:
        super().__init__(*edges, **kwargs)

    @override
    def add_edge(self, from_v: T, to_v: T, weight: Optional[int] = None)->None:
        super().add_edge(from_v, to_v, weight)
        super().add_edge(to_v, from_v, weight)

    @override
    def delete_edge(self, from_v: T, to_v: T):
        super().delete_edge(from_v, to_v)
        super().delete_edge(to_v, from_v)

    def degree(self, v: T)->int:
        index = self._get_index_of(v)
        assert index is not None
        assert self._adjacency_list[index] is not None
        return len(self._adjacency_list[index])

    def number_of_vertices_with_odd_degree(self)-> int:
        num = 0
        for v in self.labels.values():
            num += self.degree(v) % 2
        return num

    def r_regular(self)->Optional[int]:
        degrees = set([self.degree(v) for v in self.labels.values()])
        if len(degrees) != 1:
            return None
        return list(degrees)[0]

    def is_path(self, *nodes:T)->bool:
        assert len(nodes) >= 2, "path must contain at least two nodes"
        visited: set[T] = set()
        visited.add(nodes[0])
        traveled_edges: dict[T, list[T]] = {}

        for i in range(1, len(nodes)):
            if nodes[i] in visited and i != len(nodes) - 1:
                #print(f"visited {nodes[i]} twice")
                return False
            visited.add(nodes[i])
            if not self.exists_edge(nodes[i-1], nodes[i]):
                #print(f"path does not exist from {nodes[i-1]} to {nodes[i]}")
                return False
            traveled_edge = nodes[i] in traveled_edges.get(nodes[i-1], [])
            if traveled_edge:
                #print(f"traveled edge from {nodes[i-1]} to {nodes[i]} already")
                return False
            traveled_edges.setdefault(nodes[i], []).append(nodes[i-1])
            traveled_edges.setdefault(nodes[i-1], []).append(nodes[i])

        return True

    def is_cycle(self, *nodes:T)->bool:
        return self.is_path(*nodes, nodes[0])

    def get_edge_set(self) -> set[tuple[T, T]]:
        edge_set: set[tuple[int, int]] = set()
        for from_v, adjacent_vs in self._adjacency_list.items():
            for adjacent in adjacent_vs:
                if (from_v, adjacent) not in edge_set and (adjacent, from_v) not in edge_set:
                    edge_set.add((from_v, adjacent))
        return set((self.labels[from_i], self.labels[to_i]) for (from_i, to_i) in edge_set)
    
    def is_euler_tour(self, *nodes: T)-> bool:
        if nodes[0] != nodes[-1]:
            return False
        adjacency_list = deepcopy(self._adjacency_list)
        for i in range(1, len(nodes)):
            from_v = self._get_index_of(nodes[i-1])
            to_v = self._get_index_of(nodes[i])
            if from_v not in adjacency_list or to_v not in adjacency_list:
                return False
            if to_v not in adjacency_list[from_v]:
                return False
            adjacency_list[from_v].remove(to_v)
            adjacency_list[to_v].remove(from_v)
        traveled_all_edges = all([len(adjacent) == 0 for adjacent in adjacency_list.values()])
        return traveled_all_edges
    
    def is_hamiltonean(self, *nodes: T)->bool:
        contains_all_nodes = len(set(self._adjacency_list.keys()) - set([self._get_index_of(node) for node in nodes])) == 0
        return self.is_path(*nodes) and contains_all_nodes and nodes[0] == nodes[-1]

    @override
    def to_network_x(self, **kwargs) -> nx.Graph:
        G = nx.Graph()
        self._add_network_x_nodes(G, **kwargs)
        for edge_from, edge_to in self.get_edge_set():
            from_i = self.get_present_index_of(edge_from)
            to_i = self.get_present_index_of(edge_to)
            G.add_edge(str(edge_from), str(edge_to), **self._get_edge_properties(from_i, to_i))
        return G
