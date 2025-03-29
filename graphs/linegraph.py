from typing import Generic, TypeVar
from graphs.graph import Graph
from dataclasses import dataclass

T = TypeVar("T")

@dataclass
class LineGraphVertex(Generic[T]):
    from_original: T
    to_original: T


    def __hash__(self):
        return hash(frozenset((self.from_original, self.to_original)))

    def __eq__(self, other: object):
        if not isinstance(other, LineGraphVertex):
            return False
        return frozenset((self.from_original, self.to_original)) == frozenset((other.from_original, other.to_original)) # type: ignore

    def intersects(self, other: "LineGraphVertex[T]")->bool:
        s1 = set([self.from_original, self.to_original])
        s2 = set([other.from_original, other.to_original])
        return not s1.isdisjoint(s2)

    def __str__(self) -> str:
        return f"{self.from_original}{self.to_original}"


def get_line_graph[T](original: Graph[T])->Graph[LineGraphVertex[T]]:
    edges_of_new_graph: list[tuple[LineGraphVertex[T], LineGraphVertex[T]]] = []
    edge_set = original.get_edge_set()
    vertices = [LineGraphVertex(from_v, to_v) for (from_v, to_v) in edge_set]
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            if vertices[i].intersects(vertices[j]):
                edges_of_new_graph.append((vertices[i], vertices[j]))
    return Graph(*edges_of_new_graph)
