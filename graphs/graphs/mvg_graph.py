import pandas as pd
from graphs.graph import Graph

def get_edges(constant_weights: bool)->list[tuple[str, str, float]]:
    edges: list[tuple[str, str, float]] = []
    data = pd.read_csv("graphs/graphs/graph.csv")
    data = data.groupby(['from', 'to']).agg({'weight': 'mean', "line": "first"}).reset_index()
    for _, row in data.iterrows():
        weight = 1
        if not constant_weights:
            weight = float(row["weight"])
        edges.append((row["from"], row["to"], weight))
    return edges

class MvgGraphConstantWeight(Graph):
    def __init__(self) -> None:
        super().__init__(*get_edges(True))

class MvgGraph(Graph):
    def __init__(self) -> None:
        super().__init__(*get_edges(False))
