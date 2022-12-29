from constraint import Constraint
from ..utils.types import DataListType
import pandas as pd

class ConstraintNumericalRange(Constraint):
    def __init__(self, constraint_description: str = 'Range constraint') -> None:
        super().__init__(constraint_description)

    def setRange(self, lower: float = None, upper: float = None, type: type = int) -> None:
        self.lower = lower
        self.upper = upper
        self.type = type

    def checkIfSatisfied(self, data: DataListType) -> bool:
        '''
        Check if data is between lower and upper bound (both included).
        '''
        if isinstance(data, pd.Series):
            if self.lower:
                if data[data < self.lower].any():
                    return False
            if self.upper:
                if data[data > self.upper].any():
                    return False
        else:
            for number in data:
                if self.lower and number < self.lower:
                    return False
                if self.upper and number > self.upper:
                    return False
        return True


        