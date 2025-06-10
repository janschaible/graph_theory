import os
import sys
from typing import Generic, TypeVar, Optional, TypeAlias, Union
from itertools import product
import numpy as np
import networkx as nx
import pathlib

from graphs.tree import Tree
from disjoint_set import DisjointSet

T = TypeVar("T")

EdgeDefinition: TypeAlias = Union[tuple[T, T], tuple[T, T, float], tuple[T, T, float, float]]

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
        self.capacities: dict[int, dict[int, float]] = {}
        self.weighted = None

        for v in kwargs.get("vertices", []):
            self.add_vertex(v)

        for edge in edges:
            weight = None
            capacity = None
            if len(edge) == 2:
                from_v, to_v = edge
                self.weighted = False
            elif len(edge) == 3:
                from_v, to_v, weight = edge
                self.weighted = True
            else:
                from_v, to_v, weight, capacity = edge
                self.weighted = True

            self.add_edge(from_v, to_v, weight, capacity)

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
        # todo js this disregards any sinks fix this
        return list(self.get_adjacency_list().keys())

    def add_edge(self, from_v: T, to_v: T, weight: Optional[float]=None, capacity: Optional[float]=None)->None:
        from_index = self._get_or_create_vertex(from_v)
        to_index = self._get_or_create_vertex(to_v)
        assert to_index not in self._adjacency_list[from_index], f"edge from {from_v} to {to_v} is already present"
        self._adjacency_list[from_index].append(to_index)
        
        assert weight is not None if self.weighted else weight is None, f"cannot use weight of {weight} either all edges have weights or none of them"
        if weight is not None:
            self.weights.setdefault(from_index, {})[to_index] = weight
        if capacity is not None:
            self.capacities.setdefault(from_index, {})[to_index] = capacity

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

    def get_indices(self) -> set[int]:
        indices = set(self._adjacency_list.keys())
        for adjacent in self._adjacency_list.values():
            for v in adjacent:
                indices.add(v)
        return indices
    
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
        closeness_centralities = []
        if kwargs.get("eigen_centrality", False):
            eigen_centralities = self._get_eigenvalue_centralities()
        if kwargs.get("closeness_centrality", False):
            closeness_centralities = self.closeness_centrality()

        for l in self.labels.values():
            color = None
            if kwargs.get("eigen_centrality", False):
                color = self._get_centrality_color(eigen_centralities, l)
            if kwargs.get("closeness_centrality", False):
                color =  self._get_centrality_color(closeness_centralities, l)
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
        #                               s  , t   , count
        count_paths_between: dict[tuple[int, int], int] = {(a, b): 0 for (a, b) in product(range(len(distances)), repeat=2)}
        #                                        s  , t          v  , count
        count_paths_between_using_v: dict[tuple[int, int], dict[int, int]] = {(a, b): {} for (a, b) in product(range(len(distances)), repeat=2)}

        for k in range(len(distances)):
            for i in range(len(distances)):
                if distances[i][k] + distances[k][i] < 0:
                    return self._fw_construct_negative_cycle(predecessors, i, k), None, None, None
                for j in range(len(distances)):
                    distance_via_k = distances[i][k] + distances[k][j]
                    if distance_via_k == distances[i][j]:
                        # path with equal weight
                        count_paths_between[(i,j)] += 1
                        count_paths_between_using_v[(i,j)][k] = count_paths_between_using_v[(i,j)].get(k, 0) + 1
                    if distance_via_k < distances[i][j]:
                        count_paths_between[(i,j)] = 1
                        count_paths_between_using_v[(i,j)] = {k: 1}
                        distances[i][j] = distance_via_k
                        predecessors[i][j] = predecessors[k][j]
        return np.array(distances), np.array(predecessors), count_paths_between, count_paths_between_using_v
    
    '''
    todo check gpt implementation
    def floyd_warshall(self):
        n = len(self.nodes)
        distances = self.get_adjacency_matrix(float("inf"))
        for i in range(n):
            distances[i][i] = 0

        count_paths_between = {(i, j): 1 if distances[i][j] < float("inf") and i != j else 0
                            for i, j in product(range(n), repeat=2)}
        count_paths_between_using_v = {(i, j): {} for i, j in product(range(n), repeat=2)}

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if distances[i][k] + distances[k][j] < distances[i][j]:
                        distances[i][j] = distances[i][k] + distances[k][j]
                        count_paths_between[(i, j)] = count_paths_between[(i, k)] * count_paths_between[(k, j)]
                        count_paths_between_using_v[(i, j)] = {}
                        for v in [k]:
                            if v != i and v != j:
                                count_paths_between_using_v[(i, j)][v] = count_paths_between[(i, k)] * count_paths_between[(k, j)]
                    elif distances[i][k] + distances[k][j] == distances[i][j] and distances[i][j] < float("inf") and i != j:
                        # Add alternative shortest path through k
                        count_paths_between[(i, j)] += count_paths_between[(i, k)] * count_paths_between[(k, j)]
                        for v in [k]:
                            if v != i and v != j:
                                count_paths_between_using_v[(i, j)][v] = (
                                    count_paths_between_using_v[(i, j)].get(v, 0) +
                                    count_paths_between[(i, k)] * count_paths_between[(k, j)]
                                )

        return distances, None, count_paths_between, count_paths_between_using_v
    '''

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

    def closeness_centrality(self)->list[float]:
        distances,_,_,_ = self.floyd_warshall()
        centralities = np.zeros(len(self.labels))
        for i, v in self.labels.items():
            sum_distances: float = 0
            for d in distances.T[self.get_present_index_of(v)]:
                if d != float("inf"):
                    sum_distances+=d
            centralities[i] = 1/sum_distances
        return centralities.tolist()

    def betweeness_centrality(self) -> dict[T, float]:
        _, _, count_paths_between, count_paths_between_using_v = self.floyd_warshall()
        if None in (count_paths_between, count_paths_between_using_v):
            raise ValueError("can't compute centrality for graph with negative cycles")
        
        numerators: dict[int, float] = {}
        denominators: dict[int, float] = {}

        for (s, t), s_t_by_v in count_paths_between_using_v.items(): #type:ignore
            if s == t:
                continue
            total_paths = count_paths_between[(s, t)] #type:ignore
            if total_paths == 0:
                continue
            for v, count in s_t_by_v.items():
                if v == s or v == t:
                    continue
                numerators[v] = numerators.get(v, 0) + count
                denominators[v] = denominators.get(v, 0) + total_paths

        return {
            self.labels[v]: numerators[v] / denominators[v]
            for v in numerators
            if denominators[v] > 0
        }


    def bfs(self, starting_point: T)-> tuple[Tree, int]:
        '''
        returns:
        tree -- minimal spanning tree
        int -- girth of graph
        '''
        girth = sys.maxsize
        root = Tree[T](starting_point,[], str(starting_point))
        visited: dict[int, Tree] = {self.get_present_index_of(starting_point): root}
        queue: list[Tree[T]] = [root]
        while len(queue):
            tree = queue.pop(0)
            v = self.get_present_index_of(tree.node)
            for u in self.__connects_to(v, list(visited.keys())):
                len_cycle = self.__find_len_between(tree, visited[u]) + 1
                if len_cycle < girth:
                    girth = len_cycle
            neighbours = self._adjacency_list[v]
            child_trees = {
                n: Tree[T](self.labels[n],[], f"{tree.lineage}{str(self.labels[n])}")
                for n in neighbours if n not in visited
            }
            tree.children = list(child_trees.values())
            queue += child_trees.values()
            visited |= child_trees
        return root, girth


    def __connects_to(self, v: int, possible: list[int])-> list[int]:
        connects = []
        for u in self._adjacency_list[v]:
            if u in possible:
                connects.append(u)
        return connects
    
    def __find_len_between(self, t1: Tree, t2: Tree)->int:
        # abcdefg
        # abchi
        common_prefix = os.path.commonprefix([t1.lineage, t2.lineage])
        assert len(common_prefix) >= 1
        return len(t1.lineage) + len(t2.lineage) - 2*len(common_prefix)


    def prim_jarnik(self)->dict[T, set[T]]:
        tree: dict[int, set[int]] = {}
        weights = {v:float("inf") for v in self.get_indices()}
        weights[0] = 0
        predecessors: dict[int, int] = {}
        while len(weights)>0:
            j = min(weights.keys(), key=lambda k: weights[k])
            del weights[j]
            if j != 0:
                from_v = predecessors[j]
                vs = tree.get(j, set())
                vs.add(j)
                tree[from_v] = vs

            for connected in self._adjacency_list[j]:
                w = self.weights[j][connected]
                if connected in weights and w < weights[connected]:
                    weights[connected] = w
                    predecessors[connected] = j

        return {
            self.labels[from_v]: {self.labels[to_v] for to_v in to_vs}
            for from_v, to_vs in tree.items()
        }


    def kruskal(self)->dict[T, set[T]]:
        tree: dict[int, set[int]] = {}
        edges_with_weight = [
            (from_v, to_v, self.weights[from_v][to_v])
            for from_v, to_vs in self._adjacency_list.items()
            for to_v in to_vs
        ]
        edges_with_weight = sorted(edges_with_weight, key=lambda x: x[2])
        ds = DisjointSet.from_iterable(self.get_indices())
        for (from_v, to_v, _) in edges_with_weight:
            if ds.find(from_v) == ds.find(to_v):
                continue
            ds.union(from_v, to_v)
            connected = tree.get(from_v, set())
            connected.add(to_v)
            tree[from_v] = connected

        return {
            self.labels[from_v]: {self.labels[to_v] for to_v in to_vs}
            for from_v, to_vs in tree.items()
        }

    def ford_and_fulkerson(self, start_v: T, end_v: T):
        """maxflow algorithm"""
        s = self.get_present_index_of(start_v)
        t = self.get_present_index_of(end_v)
        f = {from_v: {to_v:0 for to_v in values.keys() } for from_v, values in self.capacities.items()}
        
        labels: dict[int, Optional[tuple[Optional[int], str, float]]] = {v: None for v in self.get_indices()}
        labels[s] = (None, "-", float("inf"))
        u: dict[int, bool] = {u: False for u in self.get_indices()}
        d: dict[int, float] = {u: float("inf") for u in self.get_indices()}

        while True:
            v = [v for v,label in labels.items() if label is not None and not u[v]].pop()

            # for every outgoing edge of v
            for w in self._adjacency_list[v]:
                if labels[w] is None and f[v][w] < self.capacities[v][w]:
                    d[w] = min(self.capacities[v][w]-f[v][w], d[v])
                    labels[w] = (v, "+", d[w])

            incoming = [u for u, values in self._adjacency_list.items() if v in values]
            # for every incoming edge of v
            for w in incoming:
                if labels[w] is None and f[w][v] > 0:
                    d[w] = min(f[w][v], d[v])
                    labels[w] = (v, "-", d[w])

            u[v] = True
            label_of_t = labels[t]
            if label_of_t is not None:
                _,_,diff = label_of_t
                w = t
                while w != s:
                    predecessor,sign,_ = labels[w] # type: ignore
                    if sign == "+":
                        f[predecessor][w] += diff # type: ignore
                    else:
                        f[predecessor][w]-= diff # type: ignore
                    w = predecessor

                # prepare for next round
                labels: dict[int, Optional[tuple[Optional[int], str, float]]] = {v: None for v in self.get_indices()}
                labels[s] = (None, "-", float("inf"))
                u: dict[int, bool] = {u: False for u in self.get_indices()}
                d: dict[int, float] = {u: float("inf") for u in self.get_indices()}
            if all([u[v] for v, label in labels.items() if label is not None]):
                break

        part_1 = [self.labels[v] for v, label in labels.items() if label is not None]
        part_2 = [self.labels[v] for v, label in labels.items() if label is None]
        return f, part_1, part_2
