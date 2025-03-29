from typing import TypeVar, override, Optional
from graphs.digraph import DiGraph
import graphviz
import pathlib

T = TypeVar("T")

class Graph(DiGraph[T]):
    def __init__(self, *edges: tuple[T, T]) -> None:
        super().__init__(*edges)

    @override
    def add_edge(self, from_v: T, to_v: T)->None:
        super().add_edge(from_v, to_v)
        super().add_edge(to_v, from_v)

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
                return False
            visited.add(nodes[i])
            if not self.exists_edge(nodes[i-1], nodes[i]):
                return False
            traveled_edge = nodes[i] in traveled_edges.get(nodes[i-1], [])
            if traveled_edge:
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
    
    @override
    def render(self, location: str):
        dot = graphviz.Graph()
        for l in self.labels.values():
            dot.node(str(l))
        
        for edge_from, edge_to in self.get_edge_set():
            dot.edge(str(edge_from), str(edge_to))

        pathlib.Path(location).parent.mkdir(parents=True, exist_ok=True) 
        dot.render(location, format="png")
