from graphs.graphs.mvg_graph import MvgGraph
from tests.graph_test import AbstractGraphTest

class TestMvgGraph(AbstractGraphTest):

    @classmethod
    def get_render_dir(cls) -> str:
        return "test_mvg_graph"

    def test_mvg_graph(self):
        self.render(MvgGraph(), "test_construction")

