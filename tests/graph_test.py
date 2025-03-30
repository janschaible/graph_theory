from graphs.digraph import DiGraph
from pathlib import Path
import shutil
import unittest
import pprint

class AbstractGraphTest(unittest.TestCase):
    @classmethod
    def get_render_dir(cls) -> str:
        raise Exception("must be implementex by sub classes")

    @classmethod
    def setUpClass(cls):
        cls.render_dir = f"./tests/test_graphs/{cls.get_render_dir()}"
        _clear_render_dir(cls.render_dir)
    
    def assert_graph_equal[T](self, graph: DiGraph[T], expected: dict[T, list[T]]):
            adjacency_list = graph.get_adjacency_list()
    
            # Convert lists to sets so that order doesn't matter
            adjacency_set = {k: set(v) for k, v in adjacency_list.items()}
            expected_set = {k: set(v) for k, v in expected.items()}
            
            assert adjacency_set == expected_set, f"expected: {pprint.pformat(expected)}, \ngot:{pprint.pformat(adjacency_list)}"
    
    def get_error_message[T](self, message: str, graph: DiGraph[T]) -> str:
         return f"{message}: {pprint.pformat(graph.get_adjacency_list())}"
         


    def render[T](self, graph: DiGraph[T], graph_name: str):
        graph.render(f"{self.render_dir}/{graph_name}")


def _clear_render_dir(render_dir):
    path = Path(render_dir)
    if path.exists() and path.is_dir():
        shutil.rmtree(path)