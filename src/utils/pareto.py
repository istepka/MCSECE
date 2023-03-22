from typing import List
from EasyMCDM.models.Pareto import Pareto
import numpy as np
import numpy.typing as npt  


def get_pareto_optimal_mask(data: npt.NDArray, optimization_direction: List[str]) -> npt.NDArray:
    '''
    `data`: scores to explore   
    `optimization direction`: list of ['max' and 'min'] of the data feature length

    return mask to the data with 1 in positions of pareto optimal datapoints 
    '''
    data_dic = {i: row for i, row in enumerate(data)}

    solver = Pareto(data_dic, verbose=False)
    result = solver.solve(indexes=list(range(data.shape[1])), prefs=optimization_direction)

    mask = np.zeros(data.shape[0])
    
    for k, v in result.items():
        if len(v['Dominated-by']) > 0:
            mask[int(k)] = False
        else:
            mask[int(k)] = True

    return mask

def get_ideal_point(data: npt.NDArray, optimization_direction: List[str], pareto_optimal_mask: npt.NDArray) -> npt.NDArray:
    '''
    `data`: scores to explore   
    `optimization direction`: list of ['max' and 'min'] of the data feature length
    `pareto_optimal_mask`: mask to the data with 1 in positions of pareto optimal datapoints 

    return ideal point of the pareto optimal points
    '''
    pareto_optimal_data = data[pareto_optimal_mask.astype(bool)]
    # ideal point is the point with the best score for each objective
    # it works in criteria space
    ideal_point = np.zeros(data.shape[1])
    
    for i, opt in enumerate(optimization_direction):
        if opt == 'max':
            ideal_point[i] = np.max(pareto_optimal_data[:, i])
        else:
            ideal_point[i] = np.min(pareto_optimal_data[:, i])
            
    return ideal_point

if __name__ == '__main__':
    get_pareto_optimal_mask(np.array([[0,1,2], [0,0,0], [1,1,1]]), ['max', 'max', 'max'])
    print(get_ideal_point(np.array([[0,1,2], [0,0,0], [1,1,1]]), ['max', 'max', 'max'], np.array([1,0,1])))