from dataclasses import dataclass
from typing import List


@dataclass
class ValueMaxDiff:
    columns: List[str]
    max_difference: float
