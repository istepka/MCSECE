import pandas as pd
import numpy as np
import numpy.typing as npt
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import logging
from tqdm import tqdm
from typing import List
from collections import defaultdict
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
    
from experiment_utils import *
from visualization_utils import *
from quick_rename import rename_explainers


pio.renderers.default = "browser"

# Seed for reproducibility
np.random.seed(42)

# Change cwd to experiments
if 'experiments' not in os.getcwd():
    os.chdir('experiments')

RESOLUTION = 15
PLOT_METRICS = ['DiscriminativePower(9)', 'K_Feasibility(3)', 'Proximity']
DATASETS = ['german', 'adult', 'compas', 'fico']
# This corresponds to folders containing countefactuals 
DATES = ['2023-03-12', '2023-03-14', '2023-03-15'] 
# This corresponds to folders containing countefactuals from different run where dice-1, cfproto-1 and wchater-1 were generated
DATES_OTHER_METHODS = ['2023-03-22', '2023-03-23']
ONLY_VALID_MODE = True
RESULTS_DIR = 'tmp_results'
RESULTS_PATH = os.path.join(os.getcwd(), RESULTS_DIR)
# Last 5 letters indicate distance metric
ADDITIONAL_METHODS = ['ideal_point_manha', 'ideal_point_eucli', 'ideal_point_cheby', 'random_choice'] 
# Last 2 letters indicate the number of counterfactuals generated
SUPPLEMENT_METHODS = ['dice-1', 'cfproto-1', 'wachter-1'] 


# INITIALIZATION
if not os.path.exists(RESULTS_PATH):
    os.mkdir(RESULTS_PATH)

# Set logging
logging.basicConfig(level=logging.DEBUG)


for dataset in DATASETS:
    logging.debug(f'Processing {dataset} dataset')

    # Load scores
    list_of_scores_df , scores_df_all, scores_test_set_indices = load_data('scores', DATES, dataset)
    logging.debug(f'Gathered scores for {len(list_of_scores_df)} instances')

    # Load valid_scores
    list_of_valid_scores_df , valid_scores_df_all, valid_scores_test_set_indices = load_data('valid_scores', DATES, dataset)
    logging.debug(f'Gathered valid_scores for {len(list_of_valid_scores_df)} instances')

    # Load counterfactuals
    list_of_counterfactuals_df , counterfactuals_df_all, cf_test_set_indices = load_data('counterfactuals', DATES, dataset)
    logging.debug(f'Gathered counterfactuals for {len(list_of_counterfactuals_df)} instances')

    # Load valid_counterfactuals
    list_of_valid_counterfactuals_df , valid_counterfactuals_df_all, valid_cf_test_set_indices = load_data('valid_counterfactuals', DATES, dataset)
    logging.debug(f'Gathered valid_counterfactuals for {len(list_of_valid_counterfactuals_df)} instances')
    
    # Load supplement data of methods outside of our ensemble framework
    sup_list_of_scores_df, sup_scores_df_all, sup_scores_test_set_indices = load_data('scores', DATES_OTHER_METHODS, dataset)
    sup_list_of_scores_df, sup_scores_df_all = rename_explainers(sup_list_of_scores_df, sup_scores_df_all)
    logging.debug(f'Gathered supplement data for {len(sup_list_of_scores_df)} instances')
    logging.debug(f'Missing indices: {set(scores_test_set_indices) - set(sup_scores_test_set_indices)}')
    sup_list_of_valid_scores_df, sup_valid_scores_df_all, sup_valid_scores_test_set_indices = load_data('valid_scores', DATES_OTHER_METHODS, dataset)
    sup_list_of_valid_scores_df, sup_valid_scores_df_all = rename_explainers(sup_list_of_valid_scores_df, sup_valid_scores_df_all)
    logging.debug(f'Gathered supplement data for {len(sup_list_of_scores_df)} instances')
    sup_list_of_counterfactuals_df, sup_counterfactuals_df_all, sup_cf_test_set_indices = load_data('counterfactuals', DATES_OTHER_METHODS, dataset)
    sup_list_of_counterfactuals_df, sup_counterfactuals_df_all = rename_explainers(sup_list_of_counterfactuals_df, sup_counterfactuals_df_all)
    logging.debug(f'Gathered supplement data for {len(sup_list_of_scores_df)} instances')
    sup_list_of_valid_counterfactuals_df, sup_valid_counterfactuals_df_all, sup_valid_cf_test_set_indices = load_data('valid_counterfactuals', DATES_OTHER_METHODS, dataset)
    sup_list_of_valid_counterfactuals_df, sup_valid_counterfactuals_df_all = rename_explainers(sup_list_of_valid_counterfactuals_df, sup_valid_counterfactuals_df_all)
    logging.debug(f'Gathered supplement data for {len(sup_list_of_scores_df)} instances')

    # Load test data - original x instances
    if 'experiments' in os.getcwd():
        test_data_path = os.path.join(os.path.pardir, 'data', f'{dataset}_test.csv')
    else:
        test_data_path = os.path.join(os.getcwd(), 'data', f'{dataset}_test.csv')


    test_dataset = pd.read_csv(test_data_path).iloc[scores_test_set_indices]
    logging.debug(f'Loaded test data for {len(test_dataset)} instances')

    # Load constraints for the dataset
    with open(os.path.join(os.path.pardir, 'data', f'{dataset}_constraints.json'), 'r') as f:
        constraints = json.load(f)
    logging.debug(f'Loaded constraints for: {constraints["dataset_shortname"]}')

    assert scores_test_set_indices == cf_test_set_indices
    assert len(list_of_scores_df) == len(list_of_counterfactuals_df) == len(test_dataset)

    continous_indices = [test_dataset.columns.get_loc(c) for c in constraints['continuous_features_nonsplit']]
    categorical_indices = [test_dataset.columns.get_loc(c) for c in constraints['categorical_features_nonsplit']]
    ranges = get_ranges(test_dataset, constraints)

    logging.debug(f'Continous indices: {continous_indices}')
    logging.debug(f'Categorical indices: {categorical_indices}')
    logging.debug(f'Ranges: {ranges}')

    test_plaus = instability(test_dataset, 0, list_of_counterfactuals_df[0].iloc[0], 
                             list_of_counterfactuals_df, ranges, continous_indices, 
                             categorical_indices)
    # Calculate example instability score
    logging.debug(f'Test instability: {test_plaus:.2f}')

        
    freeze_indices = [test_dataset.columns.get_loc(c) for c in constraints['non_actionable_features']]
    all_explainer_names = counterfactuals_df_all['explainer'].unique().tolist() + ADDITIONAL_METHODS + SUPPLEMENT_METHODS
    experiment_scores = {
        'proximity': {k: [] for k in all_explainer_names},
        'k_feasibility_3': {k: [] for k in all_explainer_names},
        'discriminative_power_9': {k: [] for k in all_explainer_names},
        'sparsity': {k: [] for k in all_explainer_names},
        'instability': {k: [] for k in all_explainer_names},
        'coverage': {k: 0 for k in all_explainer_names},
        'actionable': {k: 0 for k in all_explainer_names},
    }

    experiment2_list_of_scores = []

    # Calculate instability for all counterfactuals
    for i in tqdm(range(len(test_dataset)), desc='Calculating instability and other scores'):
        if ONLY_VALID_MODE:
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
                actionable_indices = get_actionable_indices(test_dataset.iloc[i], _i_counterfactuals, 
                                                            continous_indices, categorical_indices, 
                                                            freeze_indices)
                
                _i_counterfactuals = _i_counterfactuals.iloc[actionable_indices]
                _i_scores = _i_scores.iloc[actionable_indices]
                
                # Get counterfactual closest to ideal point
                iscores = _i_scores[['Proximity', 'K_Feasibility(3)', 'DiscriminativePower(9)']].to_numpy()
                
                # Apply normalization in each feature
                iscores = (iscores - iscores.min(axis=0)) / (iscores.max(axis=0) - iscores.min(axis=0))
                distance_metric = explainer_name[-5:]
                #logging.debug(f'Using distance metric: {distance_metric}')
                
                if 'eucli' in distance_metric: distance_metric = 'euclidean'
                if 'manha' in distance_metric: distance_metric = 'manhattan'
                if 'cheby' in distance_metric: distance_metric = 'chebyshev'
                
                pareto_mask = get_pareto_optimal_mask(iscores, ['min', 'min', 'max'])
                ideal_point = get_ideal_point(iscores, ['min', 'min', 'max'], pareto_mask)
                closest_idx = get_closest_to_optimal_point(iscores, ['min', 'min', 'max'], 
                                                           pareto_mask, ideal_point, distance_metric)
                _index = closest_idx
                
            elif explainer_name == 'random_choice':
                # Get random counterfactual from all counterfactuals
                _index = np.random.permutation(_i_scores.index)[0]
                
            elif explainer_name in SUPPLEMENT_METHODS:
                if ONLY_VALID_MODE:
                    _i_counterfactuals = pd.concat([_i_counterfactuals, sup_list_of_valid_counterfactuals_df[i]], ignore_index=True) \
                                            .reset_index(drop=True)
                    _i_scores = pd.concat([_i_scores, sup_list_of_valid_scores_df[i]], ignore_index=True)\
                                    .reset_index(drop=True)
                else:
                    _i_counterfactuals = pd.concat([_i_counterfactuals, sup_list_of_counterfactuals_df[i]], ignore_index=True) \
                                            .reset_index(drop=True)
                    _i_scores = pd.concat([_i_scores, sup_list_of_scores_df[i]], ignore_index=True) \
                                    .reset_index(drop=True)
                
                if explainer_name not in _i_scores['explainer'].unique():
                    continue
                
                _index = _i_scores[_i_counterfactuals['explainer'] == explainer_name].index[0]
                
            elif explainer_name not in _i_scores['explainer'].unique():
                continue
            
            else:
                # Get random counterfactual from particular explainer
                _index = np.random.permutation(_i_scores[_i_counterfactuals['explainer'] == explainer_name].index)[0]
                
            _cf = _i_counterfactuals.iloc[_index]
            _instability = instability(test_dataset, i, _cf, 
                                       list_of_counterfactuals_df, 
                                       ranges, continous_indices, 
                                       categorical_indices)
            experiment_scores['instability'][explainer_name].append(_instability)
            
            _sparsity = sparsity(test_dataset.iloc[i].to_numpy(), _cf.to_numpy(), continous_indices, categorical_indices)
            experiment_scores['sparsity'][explainer_name].append(_sparsity)
            
            _score = _i_scores.iloc[_index]
            experiment_scores['proximity'][explainer_name].append(_score['Proximity'])
            experiment_scores['k_feasibility_3'][explainer_name].append(_score['K_Feasibility(3)'])
            experiment_scores['discriminative_power_9'][explainer_name].append(_score['DiscriminativePower(9)'])
            experiment_scores['coverage'][explainer_name] += 1
            
            actionable = is_actionable(test_dataset.iloc[i].to_numpy(), _cf.to_numpy(), 
                                       continous_indices, categorical_indices, freeze_indices)
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
                
    logging.debug(experiment_scores)

    # build dataframe from experiment scores
    experiment1_df = pd.DataFrame(experiment_scores).round(2)
    experiment1_savepath = os.path.join(RESULTS_PATH, f'experiment1_{dataset}.csv')
    if ONLY_VALID_MODE:
        experiment1_df.to_csv(experiment1_savepath + '_valid.csv')
    else:
        experiment1_df.to_csv(experiment1_savepath + '.csv')

    _, _latex_df = generate_latex_table(experiment1_df)
    latex = pandas_to_latex(_latex_df, 
                            keep_formatting=True,
                            save_dir=RESULTS_PATH,
                            save_file=True,
                            save_name=f'experiment1_{dataset}.tex',
                            )
    logging.debug(latex)

    combinations = [(i/RESOLUTION, (RESOLUTION-i-k)/RESOLUTION, k/RESOLUTION) 
                    for i in range(0,RESOLUTION+1) 
                    for k in range(0,RESOLUTION-i+1)]

    results = []
    normalized_scores = []
    for scores in experiment2_list_of_scores:
        # normalize scores in columns
        scores = scores.copy().reset_index(drop=True)
        x = scores[PLOT_METRICS]
        x = (x - x.min()) / (x.max() - x.min()) 
        scores[PLOT_METRICS] = x
        normalized_scores.append(scores)

    for i, j, k in tqdm(combinations, desc='Calculating combinations'):
        counts = defaultdict(lambda: 0)
        
        for scores in normalized_scores:
            scores['weighted_score'] = np.sum(scores[PLOT_METRICS] * [i, -j, -k], axis=1) #[discr, feas, prox]
            idxmax = np.argmax(scores['weighted_score'])
            counts[scores['explainer'].iloc[idxmax]] += 1
        
        # best explainer
        best = max(counts.items(), key=lambda x: x[1])
        results.append((i, j, k, best[0], best[1] / sum(counts.values())))
        
            
    logging.debug(results)

    plot_tenary_visualization(results, dataset, RESULTS_PATH, ONLY_VALID_MODE)