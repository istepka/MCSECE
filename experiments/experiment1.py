import pandas as pd
import numpy as np
import numpy.typing as npt
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
from experiment_utils import load_data, get_closest_to_optimal_point, get_pareto_optimal_mask, get_ideal_point

for dataset in ['fico', 'adult', 'compas', 'german']:
    print(f'Processing {dataset} dataset')
    
    dates = ['2023-03-12', '2023-03-14', '2023-03-15']
    only_valid = True

    # Change cwd to experiments
    if 'experiments' not in os.getcwd():
        os.chdir('experiments')

    #path = os.path.join(os.getcwd(), 'experiments', dates, 'scores')
    # Load scores
    list_of_scores_df , scores_df_all, scores_test_set_indices = load_data('scores', dates, dataset)
    print(f'Gathered scores for {len(list_of_scores_df)} instances')

    # Load valid_scores
    list_of_valid_scores_df , valid_scores_df_all, valid_scores_test_set_indices = load_data('valid_scores', dates, dataset)
    print(f'Gathered valid_scores for {len(list_of_valid_scores_df)} instances')

    # Load counterfactuals
    list_of_counterfactuals_df , counterfactuals_df_all, cf_test_set_indices = load_data('counterfactuals', dates, dataset)
    print(f'Gathered counterfactuals for {len(list_of_counterfactuals_df)} instances')

    # Load valid_counterfactuals
    list_of_valid_counterfactuals_df , valid_counterfactuals_df_all, valid_cf_test_set_indices = load_data('valid_counterfactuals', dates, dataset)
    print(f'Gathered valid_counterfactuals for {len(list_of_valid_counterfactuals_df)} instances')

    # Load test data - original x instances
    if 'experiments' in os.getcwd():
        test_data_path = os.path.join(os.path.pardir, 'data', f'{dataset}_test.csv')
    else:
        test_data_path = os.path.join(os.getcwd(), 'data', f'{dataset}_test.csv')


    test_dataset = pd.read_csv(test_data_path).iloc[scores_test_set_indices]
    print(f'Loaded test data for {len(test_dataset)} instances')

    # Load constraints for the dataset
    with open(os.path.join(os.path.pardir, 'data', f'{dataset}_constraints.json'), 'r') as f:
        constraints = json.load(f)
    print(f'Loaded constraints for: {constraints["dataset_shortname"]}')

    assert scores_test_set_indices == cf_test_set_indices
    assert len(list_of_scores_df) == len(list_of_counterfactuals_df) == len(test_dataset)

    from typing import List


    def get_ranges(test_data: pd.DataFrame, constraints: dict) -> npt.NDArray:
        '''
        Get ranges for continous variables.
        '''
        mins = test_data[constraints['continuous_features_nonsplit']].to_numpy().min(axis=0)
        maxes = test_data[constraints['continuous_features_nonsplit']].to_numpy().max(axis=0)
        feature_ranges = maxes - mins
        return feature_ranges


    def heom(x: npt.NDArray, y: npt.NDArray, ranges: npt.NDArray, continous_indices: npt.NDArray, categorical_indices: npt.NDArray) -> float:
        '''
        Calculate HEOM distance between x and y. 
        X and Y should not be normalized. 
        X should be (n, m) dimensional.
        Y should be 1-D array.
        Ranges is max-min on each continous variables (order matters). 
        '''
        distance = np.zeros(x.shape[0])

        # Continous |x-y| / range
        distance += np.sum(np.abs(x[:, continous_indices].astype('float64') - y[continous_indices].astype('float64')) / ranges, axis=1)

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
        
        # x_counterfactuals = list_of_counterfactuals_df[x_index].to_numpy()
        # # calculate all pairs of distances between counterfactuals from x and x'
        # sum_of_distances = .0
        # for x_cf in x_counterfactuals:
        #     mean_distance = np.mean(heom(closest_counterfactuals, x_cf, ranges, continous_indices, categorical_indices))
        #     sum_of_distances += mean_distance
        # return sum_of_distances / len(x_counterfactuals)
        
        instability_score = np.min(heom(closest_counterfactuals, counterfactual.to_numpy(), ranges, continous_indices, categorical_indices))
        return instability_score
        
        
        

    continous_indices = [test_dataset.columns.get_loc(c) for c in constraints['continuous_features_nonsplit']]
    categorical_indices = [test_dataset.columns.get_loc(c) for c in constraints['categorical_features_nonsplit']]
    ranges = get_ranges(test_dataset, constraints)

    print(f'Continous indices: {continous_indices}')
    print(f'Categorical indices: {categorical_indices}')
    print(f'Ranges: {ranges}')

    test_plaus = instability(test_dataset, 0, list_of_counterfactuals_df[0].iloc[0], list_of_counterfactuals_df, ranges, continous_indices, categorical_indices)
    # Calculate example instability score
    print(f'Test instability: {test_plaus:.2f}')

    def sparsity(x_instance: npt.NDArray, cf_instance: npt.NDArray, continous_indices, categorical_indices) -> int:
        _sparsity = 0
        
        # Continous
        _sparsity += np.sum(~np.isclose(x_instance[continous_indices].astype('float64'), cf_instance[continous_indices].astype('float64'), atol=1e-05))
        
        # Categorical
        _sparsity += np.sum(~np.equal(x_instance[categorical_indices].astype('str'), cf_instance[categorical_indices].astype('str')))
        
        return _sparsity

    def is_actionable(x_instance: npt.NDArray, cf_instance: npt.NDArray, continous_indices, categorical_indices, freeze_indices) -> bool:
        for freeze_index in freeze_indices:
            if freeze_index in continous_indices \
                and not np.isclose(x_instance[freeze_index:freeze_index+1].astype('float64'), cf_instance[freeze_index:freeze_index+1].astype('float64'), atol=1e-05):
                return False
            if freeze_index in categorical_indices \
                and not np.equal(x_instance.astype('str')[freeze_index], cf_instance.astype('str')[freeze_index]):
                return False
        return True

    freeze_indices = [test_dataset.columns.get_loc(c) for c in constraints['non_actionable_features']]

    def get_actionable_indices(x_instance: pd.DataFrame | pd.Series, cf_instances: pd.DataFrame, continous_indices, categorical_indices, freeze_indices) -> npt.NDArray:
        actionability = []
        for _, _cf in cf_instances.iterrows():
            actionability.append(is_actionable(x_instance.to_numpy(), _cf.to_numpy(), continous_indices, categorical_indices, freeze_indices))
        return cf_instances[actionability].index

    all_explainer_names = counterfactuals_df_all['explainer'].unique().tolist() + ['ideal_point_eucli', 'ideal_point_cheby', 'random_choice']

    experiment_scores = {
        'proximity': {k: [] for k in all_explainer_names},
        'k_feasibility_3': {k: [] for k in all_explainer_names},
        'discriminative_power_9': {k: [] for k in all_explainer_names},
        'sparsity': {k: [] for k in all_explainer_names},
        'instability': {k: [] for k in all_explainer_names},
        'coverage': {k: 0 for k in all_explainer_names},
        'actionable': {k: 0 for k in all_explainer_names},
    }

    from tqdm import tqdm

    experiment2_list_of_scores = []

    # Calculate instability for all counterfactuals
    for i in tqdm(range(len(test_dataset)), desc='Calculating instability and other scores'):
        if only_valid:
            i_counterfactuals = list_of_valid_counterfactuals_df[i]
            i_scores = list_of_valid_scores_df[i]
        else:
            i_counterfactuals = list_of_counterfactuals_df[i]
            i_scores = list_of_scores_df[i]
        
        experiment2scores = pd.DataFrame(columns=['Proximity', 'K_Feasibility(3)', 'DiscriminativePower(9)', 'explainer'])
        
        for explainer_name in all_explainer_names:
            
            _i_counterfactuals = i_counterfactuals.copy(deep=True)
            _i_scores = i_scores.copy(deep=True)
            
            if 'ideal_point' in explainer_name:
                
                # Filter counterfactuals to include only actionable
                actionable_indices = get_actionable_indices(test_dataset.iloc[i], _i_counterfactuals, continous_indices, categorical_indices, freeze_indices)
                
                _i_counterfactuals = _i_counterfactuals.iloc[actionable_indices]
                _i_scores = _i_scores.iloc[actionable_indices]
                
                # Get counterfactual closest to ideal point
                iscores = _i_scores[['Proximity', 'K_Feasibility(3)', 'DiscriminativePower(9)']].to_numpy()
                
                # Apply normalization in each feature
                iscores = (iscores - iscores.min(axis=0)) / (iscores.max(axis=0) - iscores.min(axis=0))
                
                pareto_mask = get_pareto_optimal_mask(iscores, ['min', 'min', 'max'])
                ideal_point = get_ideal_point(iscores, ['min', 'min', 'max'], pareto_mask)
                
                distance_metric = 'euclidean' if 'eucli' in explainer_name else 'chebyshev'
                
                closest_idx = get_closest_to_optimal_point(iscores, ['min', 'min', 'max'], pareto_mask, ideal_point, distance_metric)
                #print(closest_idx)
                _index = closest_idx
            elif explainer_name == 'random_choice':
                # Get random counterfactual from all counterfactuals
                _index = np.random.permutation(_i_scores.index)[0]
            elif explainer_name not in _i_scores['explainer'].unique():
                continue
            else:
                #print(explainer_name)
                # Get random counterfactual from particular explainer
                _index = np.random.permutation(_i_scores[_i_counterfactuals['explainer'] == explainer_name].index)[0]
                
            _cf = _i_counterfactuals.iloc[_index]
            _instability = instability(test_dataset, i, _cf, list_of_counterfactuals_df, ranges, continous_indices, categorical_indices)
            experiment_scores['instability'][explainer_name].append(_instability)
            
            _sparsity = sparsity(test_dataset.iloc[i].to_numpy(), _cf.to_numpy(), continous_indices, categorical_indices)
            experiment_scores['sparsity'][explainer_name].append(_sparsity)
            
            _score = _i_scores.iloc[_index]
            experiment_scores['proximity'][explainer_name].append(_score['Proximity'])
            experiment_scores['k_feasibility_3'][explainer_name].append(_score['K_Feasibility(3)'])
            experiment_scores['discriminative_power_9'][explainer_name].append(_score['DiscriminativePower(9)'])
            experiment_scores['coverage'][explainer_name] += 1
            
            actionable = is_actionable(test_dataset.iloc[i].to_numpy(), _cf.to_numpy(), continous_indices, categorical_indices, freeze_indices)
            experiment_scores['actionable'][explainer_name] += int(actionable)
            
            # Create list of scores for experiment 2 by adding scores for idael point and random choice
            new_record = pd.DataFrame({
                'DiscriminativePower(9)': [_score['DiscriminativePower(9)']],
                'K_Feasibility(3)': [_score['K_Feasibility(3)']],
                'Proximity': [_score['Proximity']],
                'explainer': [explainer_name],
            })
            experiment2scores = pd.concat([experiment2scores, new_record], axis=0)
        experiment2_list_of_scores.append(experiment2scores)
        
        # average experiment scores
    for metric_name, v in experiment_scores.items():
        for explainer_name, scores in v.items():
            if metric_name in ['coverage', 'actionable']:
                experiment_scores[metric_name][explainer_name] = experiment_scores[metric_name][explainer_name] / len(test_dataset)
            else:
                experiment_scores[metric_name][explainer_name] = np.mean(scores)
    #print(f'{metric_name} {explainer_name}: {experiment_scores[metric_name][explainer_name]:.2f}')
    print(experiment_scores)

    # build dataframe from experiment scores
    experiment1_df = pd.DataFrame(experiment_scores).round(2)
    if only_valid:
        experiment1_df.to_csv(f'results/experiment1_{dataset}_valid.csv')
    else:
        experiment1_df.to_csv(f'results/experiment1_{dataset}.csv')

    from experiment_utils import pandas_to_latex
    print(pandas_to_latex(experiment1_df, keep_formatting=True))

    from collections import defaultdict
    resolution = 25
    metrics_to_consider = ['DiscriminativePower(9)', 'K_Feasibility(3)', 'Proximity']

    combinations = [(i/resolution, (resolution-i-k)/resolution, k/resolution) for i in range(0,resolution+1) for k in range(0,resolution-i+1)]

    results = []
    normalized_scores = []
    for scores in experiment2_list_of_scores:
        # normalize scores in columns
        scores = scores.copy().reset_index(drop=True)
        x = scores[metrics_to_consider]
        x = (x - x.min()) / (x.max() - x.min()) 
        scores[metrics_to_consider] = x
        normalized_scores.append(scores)

    for i, j, k in tqdm(combinations, desc='Calculating combinations'):
        counts = defaultdict(lambda: 0)
        
        for scores in normalized_scores:
            scores['weighted_score'] = np.sum(scores[metrics_to_consider] * [i, -j, -k], axis=1) #[discr, feas, prox]
            idxmax = np.argmax(scores['weighted_score'])
            counts[scores['explainer'].iloc[idxmax]] += 1
        
        # best explainer
        best = max(counts.items(), key=lambda x: x[1])
        results.append((i, j, k, best[0], best[1] / sum(counts.values())))
        
            
    print(results)

    import matplotlib.pyplot as plt
    import plotly.express as px
    # Plot plotly.figure_factory.create_ternary_contour
    from plotly.figure_factory import create_ternary_contour
    # Display plotly in browser (not in notebook)
    import plotly.io as pio
    pio.renderers.default = "browser"

    x,y,z,explainer,percentage = zip(*results)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    percentage = np.array(percentage)

    df = pd.DataFrame({
        'DiscriminativePower(9)': x,
        'K_Feasibility(3)': y,
        'Proximity': z,
        'explainer': explainer,
        'percentage': percentage,
    }).sort_values(by='explainer')

    fig = px.scatter_ternary(df, 
                            a="DiscriminativePower(9)", 
                            b="K_Feasibility(3)", 
                            c="Proximity", 
                            color="explainer", 
                            size="percentage", 
                            size_max=18, 
                            hover_name="explainer", 
                            color_continuous_scale=px.colors.sequential.Plasma,
                            # Title
                            title=f"Best Explainer for {dataset} dataset" + (" (only valid)" if only_valid else ""),
                            # Show axes arrow    
                            # Change plot size
                            height=800,
                            width=1000,
                            )


    # Save plot to png
    if only_valid:
        fig.write_image(f"results/experiment2_{dataset}_valid.png")
    else:
        fig.write_image(f"results/experiment2_{dataset}.png")
    fig.show()