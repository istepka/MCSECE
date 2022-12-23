from dataclasses import dataclass
from typing import List


@dataclass
class ValueChangeDirection:
    columns: List[int]
    direction: str
