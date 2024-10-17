from dataclasses import dataclass
from typing import List

@dataclass
class Instance:
    def __init__(self, text: List[str], label: str):
        self.text = text
        self.label = label
    
    def __iter__(self):
        yield self.text
        yield self.label

