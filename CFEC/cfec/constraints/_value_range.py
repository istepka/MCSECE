from dataclasses import dataclass
from typing import List


@dataclass
class ValueRange:
    columns: List[str]
    min_value: float
    max_value: float
