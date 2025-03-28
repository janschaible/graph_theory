from typing import Generic, TypeVar, Optional

T = TypeVar("T")

class DiGraph(Generic[T]):
    def __init__(self, *edges: tuple[T, T]) -> None:
        self._adjacency_list: dict[int, list[int]]  = {}
        self.labels: dict[int, T] = {}
        for from_v, to_v in edges:
            self.add_edge(from_v, to_v)

    def add_vertex(self, v: T)->int:
        assert self._get_index_of(v) is None, f"tired to add vertex {v} but was already present"
        index = self._get_next_free_index()
        self.labels[index] = v
        self._adjacency_list.setdefault(index, [])
        return index

    def delete_vertex(self, v: T):
        index = self._get_index_of(v)
        assert index is not None, f"Tried to delete vertex that was not there {v}"
        del self._adjacency_list[index]
        for adjacent in self._adjacency_list.values():
            if index in adjacent:
                adjacent.remove(index)

    def add_edge(self, from_v: T, to_v: T)->None:
        from_index = self._get_or_create_vertex(from_v)
        to_index = self._get_or_create_vertex(to_v)
        assert to_index not in self._adjacency_list[from_index], f"edge from {from_v} to {to_v} is already present"
        self._adjacency_list[from_index].append(to_index)

    def delete_edge(self, from_v: T, to_v: T):
        from_index = self.get_present_index_of(from_v)
        to_index = self.get_present_index_of(to_v)
        assert to_index in self._adjacency_list[from_index], f"tried to delete invalid edge {from_v} to {to_v}"
        self._adjacency_list[from_index].remove(to_index)
    
    def exists_edge(self, from_v: T, to_v: T)->bool:
        from_index = self._get_index_of(from_v)
        to_index = self._get_index_of(to_v)
        return from_index is not None and to_index in self._adjacency_list[from_index]

    def get_adjacency_list(self)->dict[T, list[T]]:
        adjacency_list:dict[T, list[T]] = {}
        for k, v in self._adjacency_list.items():
            adjacency_list[self.labels[k]] = [self.labels[adjacent] for adjacent in v]
        return adjacency_list

    def _get_or_create_vertex(self, v: T)-> int:
        present_index = self._get_index_of(v)
        if present_index is not None:
            return present_index
        return self.add_vertex(v)

    def _get_next_free_index(self) -> int:
        if len(self.labels) == 0:
            return 0
        return max(self.labels.keys()) + 1

    def get_present_index_of(self, vertex: T)->int:
        index = self._get_index_of(vertex)
        assert index is not None, f"required index {vertex} to be present"
        return index


    def _get_index_of(self, vertex: T)->Optional[int]:
        for k,v in self.labels.items():
            if v == vertex:
                return k
        return None

