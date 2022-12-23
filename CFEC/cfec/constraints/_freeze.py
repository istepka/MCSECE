from dataclasses import dataclass
from typing import List


@dataclass
class Freeze:
    columns: List[str]
