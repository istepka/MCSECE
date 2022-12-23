from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass(init=False)
class ValueNominal:
    columns: List[str]
    values: List[str]

    def __init__(self, columns: List[str], constraints: Optional[Dict[str, str]] = None):
        self.columns = columns
        self.values = list(constraints.values()) if constraints else []
