from typing import Generic, TypeVar, Optional, TypeAlias, Union
import numpy as np
import networkx as nx
import pathlib
import sys

T = TypeVar("T")

EdgeDefinition: TypeAlias = Union[tuple[T, T], tuple[T, T, float]]

class DiGraph(Generic[T]):
    def __init__(self, *edges: EdgeDefinition[T], **kwargs) -> None:
        """
        one edge is specified by:
            (from, to)
            (from, to, weight)
        """
        self._adjacency_list: dict[int, list[int]]  = {}
        self.labels: dict[int, T] = {}
        self.weights: dict[int, dict[int, float]] = {}
        self.weighted = None

        for v in kwargs.get("vertices", []):
            self.add_vertex(v)

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

    def add_edge(self, from_v: T, to_v: T, weight: Optional[float]=None)->None:
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

    def get_weight(self, from_v: T, to_v: T) -> float:
        from_i = self.get_present_index_of(from_v)
        to_i = self.get_present_index_of(to_v)
        return self._get_weight_from_index(from_i, to_i)
    
    def _get_weight_from_index(self, from_i: int, to_i: int) -> float:
        assert from_i in self.weights
        assert to_i in self.weights[from_i]
        return self.weights[from_i][to_i]

    def _get_index_of(self, vertex: T)->Optional[int]:
        for k,v in self.labels.items():
            if v == vertex:
                return k
        return None
    
    def _get_eigenvalue_centralities(self):
        adjacency_matrix = self.get_adjacency_matrix()
        eig_val, eig_vec = np.linalg.eig(adjacency_matrix)
        return np.abs(eig_vec.transpose()[np.argmax(eig_val)])

    def get_adjacency_matrix(self, default:float=0) -> np.ndarray:
        dimensions = max([*self._adjacency_list.keys(), *[k for adjacent in self._adjacency_list.values() for k in adjacent]])+1
        matrix = [[0 if i==j else default for j in range(dimensions)] for i in range(dimensions)]
        for i, adjacent in self._adjacency_list.items():
            for j in adjacent:
                if self.weighted:
                    matrix[i][j] = self._get_weight_from_index(i,j)
                else:
                    matrix[i][j] = 1
        return np.array(matrix)

    def _get_edge_properties(self, from_i:int, to_i:int):
        properties = {}
        if self.weighted:
            properties["weight"] = self._get_weight_from_index(from_i, to_i)
            properties["label"] = self._get_weight_from_index(from_i, to_i)
        return properties
    
    def _get_centrality_color(self, centralities: list[float], v: T)->str:
        centrality = centralities[self.get_present_index_of(v)]
        blue_value = int(abs(255 * (centrality-min(centralities))/(max(centralities)-min(centralities))))
        return f"#0000ff{blue_value:x}"
    
    def _add_network_x_nodes(self, G: nx.Graph, **kwargs):
        eigen_centralities = []
        if kwargs.get("eigen_centrality", False):
            eigen_centralities = self._get_eigenvalue_centralities()
        for l in self.labels.values():
            color = None
            if kwargs.get("eigen_centrality", False):
                color = self._get_centrality_color(eigen_centralities, l)
            G.add_node(str(l), style='filled',fillcolor=color)

    def to_network_x(self, **kwargs) -> nx.Graph:
        G = nx.DiGraph()
        self._add_network_x_nodes(G, **kwargs)

        for k, adjacent_list in self._adjacency_list.items():
            for adjacent in adjacent_list:
                from_v: T = self.labels[k]
                to_v: T = self.labels[adjacent]
                G.add_edge(str(from_v), str(to_v), **self._get_edge_properties(k, adjacent))
        return G

    def render(self, location: str, **kwargs):
        G = self.to_network_x(**kwargs)
        pathlib.Path(location).parent.mkdir(parents=True, exist_ok=True) 
        A = nx.nx_agraph.to_agraph(G)
        A.layout(prog="dot")
        A.draw(location)

    def dijkstra(self, from_v: T) -> dict[T, float]:
        assert self.weighted, "graph must be weighted to calculate shortest paths"
        distances = {v:float("inf") for v in self.labels.values()}
        distances[from_v] = 0
        unprocessed = list(self.labels.values())
        while len(unprocessed) > 0:
            u = min(unprocessed, key=lambda v:distances[v])
            unprocessed.remove(u)
            index = self.get_present_index_of(u)
            targets = self._adjacency_list[index]
            for target in targets:
                target_label = self.labels[target]
                edge_weight = self.get_weight(u, target_label)
                distances[target_label] = min(distances[target_label], distances[u] + edge_weight)
        return distances

    def floyd_warshall(self):
        distances = self.get_adjacency_matrix(float("inf"))
        predecessors = [
            [-1 if i==j or weight==float("inf") else i for j, weight in enumerate(row)]
            for i, row in enumerate(distances)
        ]
        for k in range(len(distances)):
            for i in range(len(distances)):
                if distances[i][k] + distances[k][i] < 0:
                    return self._fw_construct_negative_cycle(predecessors, i, k), np.array(predecessors)
                for j in range(len(distances)):
                    distance_via_k = distances[i][k] + distances[k][j]
                    if distance_via_k < distances[i][j]:
                        distances[i][j] = distance_via_k
                        predecessors[i][j] = predecessors[k][j]
        return np.array(distances), np.array(predecessors)

    def _fw_construct_negative_cycle(self, predecessors: list[list[int]], start: int, end: int)->np.ndarray:
        predecessor = predecessors[start][end]
        cycle:list[int] = [end, predecessor]
        i = 0
        while predecessor != start:
            predecessor = predecessors[start][predecessor]
            cycle.append(predecessor)
            i+=1
            if i > 5000:
                raise Exception("not found cycle")
        predecessor = start
        while predecessor != end:
            predecessor = predecessors[end][predecessor]
            cycle.append(predecessor)
            i+=1
            if i > 5000:
                raise Exception("not found cycle")
        return np.array([self.labels[v] for v in cycle])

    def excentricity(self, v: T)->float:
        distances = self.dijkstra(v)
        return max(distances.values())

    def center(self):
        center = None
        center_excentricity = float("inf")
        for v in self.labels.values():
            excentricity = self.excentricity(v)
            if excentricity<center_excentricity:
                center = v
                center_excentricity = excentricity
        return center

    def diameter(self)-> float:
        max_diameter = 0
        for v in self.labels.values():
            max_diameter = max(max_diameter, self.excentricity(v))
        return max_diameter

    def closeness_centrality(self, v: T):
        distances,_ = self.floyd_warshall()
        sum_distances: float = 0
        for d in distances.T[self.get_present_index_of(v)]:
            if d != float("inf"):
                sum_distances+=d
        return 1/sum_distances

    def betweeness_centrality(self, v: T):
        pass

