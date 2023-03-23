from typing import List, Tuple
import numpy as np
import numpy.typing as npt
from EasyMCDM.models.Pareto import Pareto
import pandas as pd
import os

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def chebyshev_distance(a, b):
    return np.max(np.abs(a - b))

def manhattan_distance(a, b):
    return np.sum(np.abs(a-b))

def calculate_distance_to_ideal_point(data: npt.NDArray, ideal_point: npt.NDArray, distance_metric: str = 'euclidean') -> npt.NDArray:
    """
    `data`: scores to explore
    `ideal_point`: ideal point of the pareto optimal points
    `distance_metric`: distance metric to use, either `euclidean`, 'manhattan' or `chebyshev`

    return distance of each data point to the ideal point
    """
    distance = np.zeros(data.shape[0])

    for i, ins in enumerate(data):
        if distance_metric == 'chebyshev':
            distance[i] = chebyshev_distance(ins, ideal_point)
        elif distance_metric == 'manhattan':
            distance[i] = manhattan_distance(ins, ideal_point)
        else:
            distance[i] = euclidean_distance(ins, ideal_point)

    return distance

def get_closest_to_optimal_point(data: npt.NDArray, 
                                 optimization_direction: List[str], 
                                 pareto_optimal_mask: npt.NDArray, 
                                 ideal_point: npt.NDArray,
                                 distance_metric: str = 'euclidean'
                                 ) -> npt.NDArray:
    """
    `data`: scores to explore
    `optimization direction`: list of ['max' and 'min'] of the data feature length
    `pareto_optimal_mask`: mask to the data with 1 in positions of pareto optimal datapoints
    `ideal_point`: ideal point of the pareto optimal points
    `distance_metric`: distance metric to use, either `euclidean`, 'manhattan' or `chebyshev
    
    return index of the closest point to the ideal point
    """
    distances = calculate_distance_to_ideal_point(data, ideal_point, distance_metric)
    closest_index = np.argmin(distances)
    return closest_index

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


def load_data(type: str, dates: List[str], dataset_name: str) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    `type`: either **scores** or **counterfactuals**
    `dates`: date of the experiment in format yyyy-mm-dd
    `dataset_name`: name of the dataset e.g. adult

    return list of scores/cfs for each instance, all combined scores/cfs dataframe, list of instance indexes in test data
    """
    dfs: List = [] 

    idcs = []
    
    for date in dates:
        if 'experiments' in os.getcwd():
            directory = f'./{date}'
        else:
            directory = f'./experiments/{date}'
    
        _, _, stats_filenames = os.walk(f'{directory}/{type}/').__next__()
        for stat_filename in stats_filenames:
            if dataset_name not in stat_filename:
                continue 

            dataset_name, explained_model, index, date = stat_filename.split('_')
            index = index[1:]
            date = date[:-5]
            #print(f'dataset: {dataset_name}, model: {explained_model}, index: {index}, date: {date}')
            idcs.append(int(index))
            with open(f'{directory}/{type}/' + stat_filename, 'r') as f:
                df = pd.read_csv(f)

            dfs.append(df)

    _all = pd.concat(dfs)
    return dfs, _all, idcs


# pandas dataframe to latex table
def pandas_to_latex(df: pd.DataFrame, 
                    keep_formatting: bool = True, 
                    save_file: bool = False, 
                    save_dir: str = 'experiments',
                    save_name: str = 'table.txt',
                    ) -> str:
    """Converts a pandas dataframe to a latex table.
    Args:
        df: The dataframe to convert.
        keep_formatting: Whether to keep the formatting of the dataframe.
    Returns:
        The latex table as a string.
    """
    latex = df.to_latex()
    # Replace \font-weightbold with proper latexbf formatting
    latex = latex.replace(r"\font-weightbold", r"\bfseries")
    # Insert \hline after each newline 
    latex = latex.replace(r"\\", r"\\ \hline")
    # Insert \hline at the top after first newline \n
    latex = latex.replace(r"rrr}", r"rrr} \hline")
    # Replace undersores with dashes
    latex = latex.replace("_", "-")
    # Insert bold line before ideal-point-eucli
    latex = latex.replace("ideal-point-", r"\bfseries ideal-point-")
    # Rename columns according to dictionary
    shortnames = {
        'proximity': 'prox',
        'k-feasibility-3': 'feas-3',
        'discriminative-power-9': 'discrpow-9',
        'sparsity': 'spars',
        'instability': 'plausib',
        'coverage': 'cover',
        'actionable': 'actionab',
    }
    uparrow = ['discrpow-9', 'actionab', 'cover']
    for k, v in shortnames.items():
        latex = latex.replace(f'{k} ', rf'{v} $\uparrow$' if v in uparrow else rf'{v} $\downarrow$')
    
    if save_file:
        save_path = os.path.join(save_dir, save_name)
        with open(save_path, 'w') as f:
            f.write(latex)
    
    return latex

def generate_latex_table(experiment_df: pd.DataFrame):
    '''
    Generate latex table from experiment dataframe
    '''
    
    def _highlight_top3(s, max_metric = ['discriminative_power_9', 'coverage', 'actionable']):
        #print(s)
        if s.name in max_metric:
            top = sorted(s, reverse=True)[:3]
        else:
            top = sorted(s)[:3]
        return ['font-weight: bold' if v  in top else '' for v in s]

    # bold top 3 in each metric
    res = experiment_df.style.apply(_highlight_top3, axis=0)
    # Round to 2 decimals
    res = res.format(precision=2)
    
    latex = pandas_to_latex(res, keep_formatting=True)
    return latex


def get_ranges(test_data: pd.DataFrame, constraints: dict) -> npt.NDArray:
    '''
    Get ranges for continous variables.
    '''
    mins = test_data[constraints['continuous_features_nonsplit']].to_numpy().min(axis=0)
    maxes = test_data[constraints['continuous_features_nonsplit']].to_numpy().max(axis=0)
    feature_ranges = maxes - mins
    return feature_ranges


def heom(x: npt.NDArray, y: npt.NDArray, 
         ranges: npt.NDArray, continous_indices: npt.NDArray, 
         categorical_indices: npt.NDArray) -> float:
    '''
    Calculate HEOM distance between x and y. 
    X and Y should not be normalized. 
    X should be (n, m) dimensional.
    Y should be 1-D array.
    Ranges is max-min on each continous variables (order matters). 
    '''
    distance = np.zeros(x.shape[0])

    # Continous |x-y| / range
    distance += np.sum(np.abs(x[:, continous_indices].astype('float64') 
                              - y[continous_indices].astype('float64')) / ranges
                       , axis=1)

    # Categorical - overlap
    distance += np.sum(~np.equal(x[:, categorical_indices], y[categorical_indices]), axis=1)

    return distance

def instability(test_data: pd.DataFrame, 
                x_index: int, 
                counterfactual: pd.DataFrame | pd.Series, 
                list_of_counterfactuals_df: List[pd.DataFrame], 
                ranges: npt.NDArray, 
                continous_indices: npt.NDArray | List[float], 
                categorical_indices: npt.NDArray | List[float]
                ):
    '''
    Calculate instability of counterfactuals - distance to the closest
    counterfactuals of the closest instance to original_x in test_data.
    '''
    # Find closest instance to original_x in test_data
    n = len(test_data)
    x = test_data.iloc[0:n+1].to_numpy()
    y = test_data.iloc[x_index].to_numpy()

    all_distances = heom(x, y, ranges, continous_indices, categorical_indices)
    # find closest instance to original_x in test_data
    sorting_indices = np.argsort(all_distances)
    # we do not take 0 because it is the same instance as original_x
    closest_index = np.array(list(zip(range(n), all_distances)))[sorting_indices][1][0].astype(int)
    # counterfactuals of closest x' to x
    closest_counterfactuals = list_of_counterfactuals_df[closest_index].to_numpy()
    
    instability_score = np.min(heom(closest_counterfactuals, 
                                    counterfactual.to_numpy(), 
                                    ranges, 
                                    continous_indices, 
                                    categorical_indices))
    return instability_score

def sparsity(x_instance: npt.NDArray, 
             cf_instance: npt.NDArray, 
             continous_indices, 
             categorical_indices
             ) -> int:
    '''
    Get sparsity which is the number of features changed.
    '''
    _sparsity = 0
    
    # Continous
    _sparsity += np.sum(~np.isclose(x_instance[continous_indices].astype('float64'), cf_instance[continous_indices].astype('float64'), atol=1e-05))
    
    # Categorical
    _sparsity += np.sum(~np.equal(x_instance[categorical_indices].astype('str'), cf_instance[categorical_indices].astype('str')))
    
    return _sparsity

def is_actionable(x_instance: npt.NDArray, 
                  cf_instance: npt.NDArray, 
                  continous_indices, 
                  categorical_indices, 
                  freeze_indices
                  ) -> bool:
    '''
    Test if counterfactual is actionable.
    '''
    for freeze_index in freeze_indices:
        if freeze_index in continous_indices \
            and not np.isclose(x_instance[freeze_index:freeze_index+1].astype('float64'), cf_instance[freeze_index:freeze_index+1].astype('float64'), atol=1e-05):
            return False
        if freeze_index in categorical_indices \
            and not np.equal(x_instance.astype('str')[freeze_index], cf_instance.astype('str')[freeze_index]):
            return False
    return True

def get_actionable_indices(x_instance: pd.DataFrame | pd.Series, 
                           cf_instances: pd.DataFrame, 
                           continous_indices, 
                           categorical_indices, 
                           freeze_indices
                           ) -> npt.NDArray:
    '''
    For a given instance, get indices of actionable counterfactuals among cf_instances.
    '''
    actionability = []
    for _, _cf in cf_instances.iterrows():
        actionability.append(is_actionable(x_instance.to_numpy(), _cf.to_numpy(), continous_indices, categorical_indices, freeze_indices))
    return cf_instances[actionability].index
    

if __name__ == '__main__':
    # print(os.getcwd())
    # tscores, all, test_indices = load_data('scores', ['2023-03-15', '2023-03-14', '2023-03-12'], 'fico')
    # print(len(tscores))
    
    print(generate_latex_table(pd.read_csv('experiments/results/experiment1_fico_valid.csv', index_col=0)))
    
    # for instance_scores in tscores:
    #     iscores = instance_scores[['Proximity', 'K_Feasibility(3)', 'DiscriminativePower(9)']].to_numpy()
        
    #     # Apply normalization in each feature
    #     iscores = (iscores - iscores.min(axis=0)) / (iscores.max(axis=0) - iscores.min(axis=0))
        
    #     pareto_mask = get_pareto_optimal_mask(iscores, ['min', 'min', 'max'])
    #     ideal_point = get_ideal_point(iscores, ['min', 'min', 'max'], pareto_mask)
    #     closest_idx = get_closest_to_optimal_point(iscores, ['min', 'min', 'max'], pareto_mask, ideal_point, 'euclidean')
        
    #     print(f'Closest idx: {closest_idx}')
    #     print(f'Best counterfactual: {instance_scores.iloc[closest_idx]}')
        
    #     vis = np.concatenate([iscores, ideal_point.reshape(1, -1)], axis=0)
    #     vis_df = pd.DataFrame(vis, columns=['Proximity', 'K_Feasibility(3)', 'DiscriminativePower(9)'])
    #     color = np.array(['dominated'] * vis_df.shape[0])
    #     color[pareto_mask.astype(bool).tolist() + [False]] = 'pareto'
    #     color[-1] = 'ideal'
    #     color[closest_idx] = 'closest'
    #     vis_df['type'] = color
        
    #     # Visualize using plotly express and pio in browser
    #     import plotly.express as px
    #     import plotly.io as pio
        
    #     pio.renderers.default = "browser"
    #     fig = px.scatter_3d(vis_df, x='Proximity', y='K_Feasibility(3)', z='DiscriminativePower(9)', color='type')
    #     fig.show()
        
    #     break
        
    