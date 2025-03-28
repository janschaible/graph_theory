from typing import TypeVar, override, Optional
from graphs.digraph import DiGraph

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
