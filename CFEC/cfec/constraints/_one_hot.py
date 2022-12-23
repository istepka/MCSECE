from dataclasses import dataclass


@dataclass
class OneHot:
    name: str
    start_column: int
    end_column: int
