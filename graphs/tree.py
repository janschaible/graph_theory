from typing import TypeVar
from dataclasses import dataclass

T = TypeVar("T")

@dataclass
class Tree[T]:
    node: T
    children: list["Tree[T]"]
    lineage: str

    def to_str(self, indent: int)->str:
        str_val = f"{'  ' * indent}{self.node}({self.lineage}){":" if len(self.children) else ''}\n"
        
        for c in self.children:
            str_val += c.to_str(indent+1)
        return str_val

    def __str__(self) -> str:
        return self.to_str(0)
