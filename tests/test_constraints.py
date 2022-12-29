import unittest
from src.constraints.constraint_numerical_range import ConstraintNumericalRange
from src.constraints.constraint_numerical_range import ConstraintNumericalRange

import pandas as pd

class TestConstraints(unittest.TestCase):
    
    def test_ConstraintNumericalRange(self):
        constraint = ConstraintNumericalRange()
        constraint.setRange(0, 100)

        data = [0, 1, 2]
        data_neg = [-1, 10]
        data_series = pd.Series(data)
        data_series_neg = pd.Series(data_neg)

        self.assertTrue(constraint.checkIfSatisfied(data))
        self.assertTrue(constraint.checkIfSatisfied(data_neg))
        self.assertTrue(constraint.checkIfSatisfied(data_series))
        self.assertTrue(constraint.checkIfSatisfied(data_series_neg))
