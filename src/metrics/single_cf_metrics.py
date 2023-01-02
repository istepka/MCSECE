from typing import Union
import numpy.typing as npt
import numpy as np
from .distance_functions import getL1distanceFloat, getL2distanceFloat, getL1distanceMatrix, getL2distanceMatrix


def validity(x_class: int | float, cf_class: int | float) -> bool:
    '''Check if x class is different than cf class. Return true if they are different.'''
    return int(x_class) != int(cf_class)

def actionability(x: npt.ArrayLike, cf: npt.ArrayLike, mask: npt.ArrayLike) -> bool: 
    '''Check if changes are allowed by mask of non-actionable features (1 change allowed, 0 not allowed)'''
    # Check if x - cf with mask applied is the same as without the mask. 
    # If not then non-actionable feature is changed making the counterfactual incorrect 
    return len(mask) == np.sum((x - cf) * mask == (x - cf)) 

def proximity(x: npt.ArrayLike, cf: npt.ArrayLike) -> float:
    '''Get L1 proximity measure between x and cf'''
    return getL1distanceMatrix(x, cf)

def features_changed(x: npt.ArrayLike, cf: npt.ArrayLike) -> int:
    '''Get the number of features that differ between x and cf'''
    return np.sum(x != cf)

def simplicity(x: npt.ArrayLike, cf: npt.ArrayLike) -> int:
    '''
    Get the number of features that differ between x and cf. 
    This function just calls features_changed function.
    '''
    return features_changed(x, cf)

def feasibility(x: npt.ArrayLike, cf: npt.ArrayLike, training_data: npt.ArrayLike) -> float:
    '''
    Shortest euclidean (L2) distance between counterfactual `cf` and intance `z` from training data where `z` != `x` 
    '''
    shortest = np.inf
    for z in training_data:
        if not np.array_equal(z, x):
            shortest = min(shortest, getL2distanceMatrix(cf, z))
    return shortest

def discriminative_power(x: npt.ArrayLike, cf: npt.ArrayLike, training_data: npt.ArrayLike, training_data_labels: npt.ArrayLike, k: int = 10) -> float:
    '''NOT IMPLEMENTED Reclassification power of k nearest neighbors from `training data`.'''
    NotImplementedError

def preference_dcg_score(x: npt.ArrayLike, cf: npt.ArrayLike, preference: npt.ArrayLike, k: int = 5) -> float:
    '''
    Calculate preference score as DCG@k.  # TODO NOT SURE IF FORMULA IS CORRECT 
    
    Parameters:  
        `preference`: list of indices that is interpreted as preference order.   
        If features have the same preference weight then they should be added as list at respective index.  
        e.g. preference = [5, 2, 6, 7, 10] and  k = 5 would mean that changes on feature with index 5 are preferred the most.
    '''
    assert k == len(preference), "Parameter k should be equal to preference list length"
    changes = x - cf
    score = 0.0

    # DCG
    # TODO NOT SURE IF FORMULA IS CORRECT 
    for i in range(1, k+1):
        score += abs(changes[i]) / np.log2(i + 1)
    
    return score

def preference_precision_score(x: npt.ArrayLike, cf: npt.ArrayLike, preference: npt.ArrayLike, k: int = 5) -> float:
    '''
    Calculate preference score as precision@k = number_of_relevant_feature_changes_within_top_k.
    '''
    assert k == len(preference), "Parameter k should be equal to preference list length"
    changes = x != cf
    count = 0
    for i in preference:
        if changes[i] == 1:
            count += 1
    return count / k
            


    

if __name__ == '__main__':

    print('Test validity')
    print(validity(0, 0))
    print(validity(1, 0))

    print('Test actionability')
    print(actionability(np.array([1,1,0]), np.array([1,1,1]), np.array([0,0,1])))
    print(actionability(np.array([1,0,0]), np.array([1,1,1]), np.array([0,0,1])))

    print('Test proximity')
    print(proximity(np.array([1,1,0]), np.array([1,1,1])))
    print(proximity(np.array([1,1,0]), np.array([1,5,5])))

    print('Test features changed')
    print(features_changed(np.array([1,1,0]), np.array([1,1,1])))
    print(features_changed(np.array([1,1,1]), np.array([1,1,1])))

    print('Test feasibility')
    print(feasibility(
        np.array([1,1,0]),
        np.array([1,1,1]),
        np.array([
            np.array([0,0,0]),
            np.array([1,0,0]),
            np.array([1,1,0])
        ])
    ))

    print('Test preference dcg score')
    print(preference_dcg_score(
        np.array([0, 3, 5, 6]),
        np.array([1, 4, 5, 6]),
        np.array([0, 2, 1]),
        k = 3
    ))

    print('Test preference precision score')
    print(preference_precision_score(
        np.array([0, 3, 5, 6]),
        np.array([1, 4, 5, 6]),
        np.array([0, 2, 1]),
        k = 3
    ))

