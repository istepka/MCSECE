import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
from typing import Dict, List, Tuple
from collections import defaultdict

from experiment_utils import load_stats


DATASETS = ['german', 'adult', 'compas', 'fico']
DATES = ['2023-03-12', '2023-03-14', '2023-03-15']

EXPLAINERS = ['cem', 'face', 'growing-spheres', 'cfproto', 'cadex', 'wachter', 'fimap', 'actionable-recourse', 'dice']
COLLECT_STAT = ['all_cfs_count', 'valid_cfs_count', 'valid_actionable_cfs_count', 'pareto_frontier_count', 'cover']

WRITE_DIR = os.path.join(os.getcwd(), 'experiments', 'results', 'experiment1_stats')
if os.path.exists(WRITE_DIR) is False:
    os.makedirs(WRITE_DIR)

aggregated = {
    'german': {k: {kk: 0 for kk in COLLECT_STAT} for k in EXPLAINERS},
    'adult': {k: {kk: 0 for kk in COLLECT_STAT} for k in EXPLAINERS},
    'compas': {k: {kk: 0 for kk in COLLECT_STAT} for k in EXPLAINERS},
    'fico': {k: {kk: 0 for kk in COLLECT_STAT} for k in EXPLAINERS},
}

for dataset in DATASETS:
    
    stats, idcs = load_stats(DATES, dataset)
    n = len(stats)
    for dic in stats:
        if dic['dataset'] != dataset:
            continue
        for explainer, explainer_data in dic['explainers'].items():
            for stat in COLLECT_STAT:
                if stat == 'cover' and explainer_data['all_cfs_count'] > 0:
                    aggregated[dataset][explainer][stat] += 1
                else:
                    aggregated[dataset][explainer][stat] += explainer_data[stat]
        
            

    for explainer, explainer_data in aggregated[dataset].items():
        for stat in COLLECT_STAT:
            aggregated[dataset][explainer][stat] /= n
            
for dataset, data in aggregated.items():
    df = pd.DataFrame(data).transpose().round(2)
    df = df.rename(columns={
        'all_cfs_count': 'all',
        'valid_cfs_count': 'val',
        'valid_actionable_cfs_count': 'act',
        'pareto_frontier_count': 'front',
        'cover': 'cov',
    }).drop(columns=['cov'])
    latex = df.to_latex()
    latex = latex.replace(r'\\', r'\\ \hline')
    latex = latex.replace('actionable-recourse', 'ar')
    latex = latex.replace('growing-spheres', 'gs')
    latex = latex.replace(rf'\toprule', '\hline')
    latex = latex.replace(rf'\bottomrule', '\hline')
    latex = latex.replace(rf'\midrule', '\hline')
    with open(os.path.join(WRITE_DIR, f'{dataset}_stats.tex'), 'w') as f:
        f.write(latex)

print(aggregated)
print(EXPLAINERS)
