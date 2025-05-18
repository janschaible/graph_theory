import pandas as pd
from graphs.graph import Graph

def get_edges()->list[tuple[str, str, float]]:
    edges: list[tuple[str, str, float]] = []
    data = pd.read_csv("graphs/graphs/graph.csv")
    data = data.groupby(['from', 'to']).agg({'weight': 'mean', "line": "first"}).reset_index()
    for _, row in data.iterrows():
        edges.append((row["from"], row["to"], float(row["weight"])))
    return edges

class MvgGraph(Graph):
    def __init__(self) -> None:
        super().__init__(*get_edges())

