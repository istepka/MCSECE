from src.metrics.single_cf_metrics import *
import pytest

def test_validity():
    assert validity(0, 0) == False
    assert validity(1, 0) == True

def test_actionability():
    assert actionability(np.array([1,1,0]), np.array([1,1,1]), np.array([0,0,1])) == 1
    assert actionability(np.array([1,0,0]), np.array([1,1,1]), np.array([0,0,1])) == 0

def test_proximity():
    assert proximity(np.array([1,1,0]), np.array([1,1,1])) == 1
    assert proximity(np.array([1,1,0]), np.array([1,5,5])) == 9

def test_features_changed():
    assert features_changed(np.array([1,1,0]), np.array([1,1,1])) == 1
    assert features_changed(np.array([1,1,1]), np.array([1,1,1])) == 0

def test_feasibility():
    assert feasibility(
        np.array([1,1,0]),
        np.array([1,1,1]),
        np.array([
            np.array([0,0,0]),
            np.array([1,0,0]),
            np.array([1,1,0])
        ])
    ) == 2

def test_discriminative_power():
    assert discriminative_power(
        np.array([1,1,0,0]),
        0,
        np.array([0,1,1,1]),
        1,
        np.array([
            np.array([0,0,0,0]),
            np.array([1,0,0,0]),
            np.array([1,1,0,1]),
            np.array([0,1,1,0]),
            np.array([0,0,1,1]),
            np.array([0,1,0,1]),
        ]),
        np.array([0, 0, 0, 1, 1, 1]),
        k = 4
    ) == 0.5

def test_preference_dcg_score():
    assert preference_dcg_score(
        np.array([0, 3, 5, 6]),
        np.array([1, 4, 5, 6]),
        np.array([0, 2, 1]),
        k = 3
    ) == 1.0

def test_preference_precision_score():
    assert round(preference_precision_score(
        np.array([0, 3, 5, 6]),
        np.array([1, 4, 5, 6]),
        np.array([0, 2, 1]),
        k = 3
    ),3) == round(0.666666, 3)


