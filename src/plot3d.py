import pandas as pd
import matplotlib.pyplot as plt
from pareto import get_pareto_optimal_mask
import numpy as np

explained_model_name = 'NN' # 'RF' / 'NN'
dataset_name = 'adult'
instance_to_explain_index = 66

def get_optimization_direction(metric_name: str) -> str:
    cost_criteria = ['feasibility', 'proximity', 'features']
    gain_criteria = ['discriminative', 'dcg']

    cost = any([True if x.lower() in metric_name.lower() else False for x in cost_criteria])
    if cost:
        return 'min'
    else:
        return 'max'

scores_to_plot = pd.read_csv(f'src/cf_scores_tmp-{explained_model_name}-{dataset_name}-{instance_to_explain_index}.csv')

counts = scores_to_plot['explainer'].value_counts()

colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w']
markers = ['s', 'o', 'v', '+', '*', 'p', 'P', 'X', 'D', '>']


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')

# 'Discriminative Power k=20', Features Changed (normalized), DCG @6, Feasibility

metric = 'Proximity'
other_metric = 'Features Changed (normalized)'
other_other_metric = 'Discriminative Power k=20'

all_x = scores_to_plot[metric].to_numpy()
all_y = scores_to_plot[other_metric].to_numpy()
all_z = scores_to_plot[other_other_metric].to_numpy()
to_check = np.array([all_x, all_y, all_z], dtype=np.float64).T

print(metric, other_metric, other_other_metric)
print(to_check)

# Get pareto frontiers mask
metric_direction = get_optimization_direction(metric)
other_metric_direction = get_optimization_direction(other_metric)
other_other_metric_direction = get_optimization_direction(other_other_metric)
optimization_directions = [metric_direction, other_metric_direction, other_other_metric_direction]
all_pareto = get_pareto_optimal_mask(data=to_check, optimization_direction=optimization_directions).astype('bool')

for plot_round in ['nonpareto', 'pareto']:
    for k, explainer in enumerate(scores_to_plot['explainer'].value_counts().sort_values(ascending=True).index.tolist()):

        mask = scores_to_plot['explainer'] == explainer
        pareto = all_pareto[mask]

        x = scores_to_plot[mask][metric].to_numpy()
        y = scores_to_plot[mask][other_metric].to_numpy()
        z = scores_to_plot[mask][other_other_metric].to_numpy()

        if plot_round == 'nonpareto':
            ax.scatter(x[~pareto], y[~pareto], z[~pareto], color='steelblue', marker=markers[k], label=explainer, alpha=0.95, s=150)
        elif plot_round == 'pareto':
            ax.scatter(x[pareto], y[pareto], z[pareto], color='orange', marker=markers[k], alpha=0.95, s=150)
         
ax.set_xlabel(f'({metric_direction}) {metric}')
ax.set_ylabel(f'({other_metric_direction}) {other_metric}')
ax.set_zlabel(f'({other_other_metric_direction}) {other_other_metric}')

plt.title(f'Pareto frontiers of the counterfactuals (lower is better)\nExplained model: {explained_model_name}\nDataset: {dataset_name}\nCounterfactuals by method {counts.to_dict()}\n')

plt.savefig(f'images/{dataset_name}/{explained_model_name}/3d_{dataset_name}_{explained_model_name}_pairplot_with_frontiers_{metric}-{other_metric}-{other_other_metric}_{instance_to_explain_index}.png')
plt.tight_layout()
plt.legend()
plt.show()