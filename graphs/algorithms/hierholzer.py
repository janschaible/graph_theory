import random
import networkx as nx
from typing import TypeVar, Optional
from copy import deepcopy
from graphs.digraph import DiGraph

T = TypeVar("T")

class Hierholzer[T]:
    def __init__(self, graph: DiGraph[T], render_path: Optional[str]=None):
        self.untraveled_graph = deepcopy(graph)
        self.cycles: list[list[T]] = []
        self.render_path = render_path
        self.render_count = 0

    def solve(self)->list[T]:
        print("== Hierholzer solver")
        self._render_untraveled_graph()
        starting_vertex = self._pick_random_untraveled_vertex()
        while starting_vertex is not None:
            self._build_cycle(starting_vertex)
            starting_vertex = self._pick_random_untraveled_vertex()
            self._render_untraveled_graph()
        solution = self._combine_cycles()
        self._render_solution(solution)
        print(f"solution {solution}")
        return solution

    def _build_cycle(self, starting_vertex: T):
        cycle = [starting_vertex]
        from_v = starting_vertex
        while True:
            possible_edges = self.untraveled_graph.get_edges_from(from_v)
            assert len(possible_edges)>0, "there is no euler cycle in the graph"
            to_v = possible_edges[0]
            self.untraveled_graph.delete_edge(from_v, to_v)
            self._drop_if_isolated(from_v)
            self._drop_if_isolated(to_v)
            from_v = to_v
            if from_v == starting_vertex:
                break
            cycle.append(to_v)

        print(f"found cycle {cycle}")
        assert len(cycle)>=2, "expected a cycle to have at least two edges"
        self.cycles.append(cycle)

    def _render_untraveled_graph(self):
        if self.render_path is None:
            return
        self.untraveled_graph.render(f"{self.render_path}/step{self.render_count}.png")
        self.render_count += 1
    
    def _render_solution(self, solution: list[T]):
        order: dict[T, dict[T, int]] = {}
        for i in range(1, len(solution)):
            from_v = solution[i-1]
            to_v = solution[i]
            order.setdefault(from_v, {})[to_v] = i
        if self.render_path is None:
            return
        G = nx.DiGraph()
        for cycle in self.cycles:
            cycle_color = random_color()
            for i in range(1, len(cycle) + 1):
                from_v = cycle[i-1]
                to_v = cycle[i % len(cycle)]
                G.add_edge(str(from_v), str(to_v), color=cycle_color, label=order[from_v][to_v])
        render_networkx(G, f"{self.render_path}/solution.png")

    def _drop_if_isolated(self, vertex: T):
        in_degree = len(self.untraveled_graph.get_edges_from(vertex))
        out_degree = len(self.untraveled_graph.get_edges_to(vertex))
        if in_degree == 0 and out_degree == 0:
            self.untraveled_graph.delete_vertex(vertex)

    def _pick_random_untraveled_vertex(self)->Optional[T]:
        remaining_vertices = self.untraveled_graph.get_vertices()
        if len(remaining_vertices) == 0:
            return None
        return remaining_vertices[0]

    def _combine_cycles(self):
        cycles_by_cross_ways = self._find_cycles_by_cross_ways()
        print(f"cross ways {cycles_by_cross_ways}")
        return self._traverse_cross_ways(0,0,cycles_by_cross_ways) + [self.cycles[0][0]]
    
    def _traverse_cross_ways(self, cycle_i: int, start_i: int, cycles_by_cross_ways: dict[T, list[int]])->list[T]:
        prune_cycle(cycles_by_cross_ways, cycle_i)
        solution: list[T] = []
        for i in range(start_i, start_i+len(self.cycles[cycle_i])):
            v = self.cycles[cycle_i][i%len(self.cycles[cycle_i])]
            print(f"{v} from cycle {cycle_i}")
            if v in cycles_by_cross_ways:
                inner_cycles = cycles_by_cross_ways[v]
                del cycles_by_cross_ways[v]
                for inner_cycle in inner_cycles:
                    solution += self._traverse_cross_ways(
                        inner_cycle,
                        self.cycles[inner_cycle].index(v),
                        cycles_by_cross_ways
                    )
                solution.append(v)
            else:
                solution.append(v)
        return solution

    def _find_cycles_by_cross_ways(self)->dict[T, list[int]]:
        element_counts: dict[T, list[int]] = {}
        for i, cycle in enumerate(self.cycles):
            unique_elements = set(cycle)
            for elem in unique_elements:
                element_counts.setdefault(elem, []).append(i)

        return {elem: cycles for elem, cycles in element_counts.items() if len(cycles) > 1}

def prune_cycle(cycles_by_cross_ways: dict[T, list[int]],cycle_i: int):
    keys = list(cycles_by_cross_ways.keys())
    for k in keys:
        if cycle_i in cycles_by_cross_ways[k]:
            cycles_by_cross_ways[k].remove(cycle_i)
            if len(cycles_by_cross_ways[k]) == 0:
                del cycles_by_cross_ways[k]

def random_color():
    return f"#{random_hex()}{random_hex()}{random_hex()}"

def random_hex():
    return f"{random.randint(0, 255):02x}"

def render_networkx(G: nx.Graph, location: str):
    A = nx.nx_agraph.to_agraph(G)
    A.layout(prog="dot")
    A.draw(location)