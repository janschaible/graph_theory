from typing import Generic, TypeVar, Optional, TypeAlias, Union
import numpy as np
import networkx as nx
import pathlib

T = TypeVar("T")

EdgeDefinition: TypeAlias = Union[tuple[T, T], tuple[T, T, int]]
EdgeMapping: TypeAlias  = dict[int, dict[int, T]]

class DiGraph(Generic[T]):
    def __init__(self, *edges: EdgeDefinition[T]) -> None:
        """
        one edge is specified by:
            (from, to)
            (from, to, weight)
        """
        self._adjacency_list: dict[int, list[int]]  = {}
        self.labels: dict[int, T] = {}
        self.weights: EdgeMapping[int] = {}
        self.weighted = None

        for edge in edges:
            if len(edge) == 2:
                from_v, to_v = edge
                weight = None
                self.weighted = False
            else:
                from_v, to_v, weight = edge
                self.weighted = True
            self.add_edge(from_v, to_v, weight)

    def add_vertex(self, v: T)->int:
        assert self._get_index_of(v) is None, f"tired to add vertex {v} but was already present"
        index = self._get_next_free_index()
        self.labels[index] = v
        self._adjacency_list.setdefault(index, [])
        return index

    def delete_vertex(self, v: T):
        index = self._get_index_of(v)
        assert index is not None, f"Tried to delete vertex that was not there {v}"
        for to_i in self._adjacency_list[index]:
            self.delete_edge_properties(index, to_i)

        del self._adjacency_list[index]
        for from_i, adjacent in self._adjacency_list.items():
            if index in adjacent:
                self.delete_edge_properties(from_i, index)
                adjacent.remove(index)
    
    def get_vertices(self)->list[T]:
        return list(self.get_adjacency_list().keys())

    def add_edge(self, from_v: T, to_v: T, weight: Optional[int]=None)->None:
        from_index = self._get_or_create_vertex(from_v)
        to_index = self._get_or_create_vertex(to_v)
        assert to_index not in self._adjacency_list[from_index], f"edge from {from_v} to {to_v} is already present"
        self._adjacency_list[from_index].append(to_index)
        
        assert weight is not None if self.weighted else weight is None, f"cannot use weight of {weight} either all edges have weights or none of them"
        if weight is not None:
            self.weights.setdefault(from_index, {})[to_index] = weight

    def delete_edge(self, from_v: T, to_v: T):
        from_index = self.get_present_index_of(from_v)
        to_index = self.get_present_index_of(to_v)
        assert to_index in self._adjacency_list[from_index], f"tried to delete invalid edge {from_v} to {to_v}"
        self._adjacency_list[from_index].remove(to_index)
        self.delete_edge_properties(from_index, to_index)

    def delete_edge_properties(self, from_i: int, to_i: int):
        if from_i in self.weights and to_i in self.weights[from_i]:
            del self.weights[from_i][to_i]

    def exists_edge(self, from_v: T, to_v: T)->bool:
        from_index = self._get_index_of(from_v)
        to_index = self._get_index_of(to_v)
        return from_index is not None and to_index in self._adjacency_list[from_index]
    
    def get_edges_from(self, from_v: T)->list[T]:
        from_index = self._get_index_of(from_v)
        if not from_index in self._adjacency_list:
            return []
        return self._resolve_indices(self._adjacency_list[from_index])

    def get_edges_to(self, to_v: T)->list[T]:
        to_index = self._get_index_of(to_v)
        from_indices = []
        for i, adjacent in self._adjacency_list.items():
            if to_index in adjacent:
                from_indices.append(i)
        return self._resolve_indices(from_indices)

    def get_adjacency_list(self)->dict[T, list[T]]:
        adjacency_list:dict[T, list[T]] = {}
        for k, v in self._adjacency_list.items():
            adjacency_list[self.labels[k]] = self._resolve_indices(v)
        return adjacency_list

    def _resolve_indices(self, indices: list[int])->list[T]:
        return [self.labels[index] for index in indices]

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

    def get_weight(self, from_v: T, to_v: T) -> int:
        from_i = self.get_present_index_of(from_v)
        to_i = self.get_present_index_of(to_v)
        return self._get_weight_from_index(from_i, to_i)
    
    def _get_weight_from_index(self, from_i: int, to_i: int) -> int:
        assert from_i in self.weights
        assert to_i in self.weights[from_i]
        return self.weights[from_i][to_i]

    def _get_index_of(self, vertex: T)->Optional[int]:
        for k,v in self.labels.items():
            if v == vertex:
                return k
        return None
    
    def to_network_x(self) -> nx.Graph:
        G = nx.Graph()
        for l in self.labels.values():
            G.add_node(str(l))
        
        for k, adjacent_list in self._adjacency_list.items():
            for adjacent in adjacent_list:
                from_v: T = self.labels[k]
                to_v: T = self.labels[adjacent]
                G.add_edge(str(from_v), str(to_v), **self._get_edge_properties(k, adjacent))
        return G

    def get_adjacency_matrix(self) -> np.ndarray:
        dimensions = max([*self._adjacency_list.keys(), *[k for adjacent in self._adjacency_list.values() for k in adjacent]])+1
        matrix = np.zeros((dimensions, dimensions))
        for i, adjacent in self._adjacency_list.items():
            for j in adjacent:
                matrix[i][j] = True
        return matrix

    def _get_edge_properties(self, from_i:int, to_i:int):
        properties = {}
        if self.weighted:
            properties["weight"] = self._get_weight_from_index(from_i, to_i)
            properties["label"] = self._get_weight_from_index(from_i, to_i)
        return properties

    def render(self, location: str):
        G = self.to_network_x()
        pathlib.Path(location).parent.mkdir(parents=True, exist_ok=True) 
        A = nx.nx_agraph.to_agraph(G)
        A.layout(prog="dot")
        A.draw(location)
