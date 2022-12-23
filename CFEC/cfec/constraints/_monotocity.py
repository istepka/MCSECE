from dataclasses import dataclass
from typing import List


@dataclass
class ValueMonotonicity:
    columns: List[str]
    direction: str
