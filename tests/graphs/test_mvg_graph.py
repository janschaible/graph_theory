import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from graphs.graphs.mvg_graph import MvgGraph, MvgGraphConstantWeight
from tests.graph_test import AbstractGraphTest

class TestMvgGraph(AbstractGraphTest):

    @classmethod
    def get_render_dir(cls) -> str:
        return "test_mvg_graph"

    def test_mvg_graph(self):
        self.render(MvgGraph(), "test_construction")
    
    def test_shortest_distances(self):
        graph = MvgGraphConstantWeight()
        matrix,_,_,_ = graph.floyd_warshall()
        self.plot_distance_matrix(matrix, graph.labels, "constant_weights", lambda x: str(round(x)))
        graph = MvgGraph()
        matrix,_,_,_ = graph.floyd_warshall()
        self.plot_distance_matrix(matrix, graph.labels, "seconds_weights", lambda x: str(round(x/60,1)))

    def test_centrality(self):
        graph = MvgGraph()
        self.render(graph, "closeness_centrality", closeness_centrality=True)

    def plot_distance_matrix(self, matrix: np.ndarray, label_map: dict[int, str], name: str, formatter: Callable[[float], str]):
        fig, ax = plt.subplots()
        ax.imshow(matrix)

        labels = [label_map[i] for i in range(len(label_map))]
        ax.set_xticks(
            range(len(matrix)),
            labels=labels,
            rotation=90,
            ha="right",
            rotation_mode="anchor",
            fontsize=2,
        )
        ax.set_yticks(range(len(matrix)), labels=labels, fontsize=2)

        max_from = ""
        max_to = ""
        max_val = 0

        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                if value > max_val:
                    max_val = value
                    max_from = labels[i]
                    max_to = labels[j]
                ax.text(
                    j,
                    i,
                    formatter(value),
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=1,
                )
        fig.suptitle(f"Max From {max_from} to {max_to} with {formatter(max_val)}")
        fig.tight_layout()
        self.save_fig(fig, name)


