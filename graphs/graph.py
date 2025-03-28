from typing import TypeVar, override
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
