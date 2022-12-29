from ..utils.types import DataListType 


class Constraint:

    def __init__(self, constraint_description: str = 'Basic constraint') -> None:
        self.constraint_description = constraint_description

    def __repr__(self) -> str:
        return self.constraint_description

    def __str__(self) -> str:
        return self.__repr__()
    
    def checkIfSatisfied(self, data: DataListType) -> bool:
        '''Check if constraint is satisfied'''
        return True

    