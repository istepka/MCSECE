from experiment_utils import load_data, get_pareto_optimal_mask, get_ideal_point
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

DATES = ['2023-03-12', '2023-03-14', '2023-03-15']
SAVE_PATH = os.path.join(os.getcwd(), 'experiments', 'results_visualization', 'pareto_front_graphs')
if os.path.exists(SAVE_PATH) is False:
    os.makedirs(SAVE_PATH)
    
# PLOT_DATASETS = ['adult', 'german', 'compas', 'fico']
PLOT_DATASETS = ['adult']

#DISTANCE_METRIC = 'manhattan' # 'euclidean', 'manhattan', 'chebyshev' 

for dataset in PLOT_DATASETS:
    list_of_scores_dfs, _, _ = load_data('valid_scores', DATES, dataset)
    for index in [48,47,49]:
        i_scores = list_of_scores_dfs[index].drop(columns=['Feasibility', 'DCG@6', 'explainer'])
        
        i_scores.rename(columns={'Proximity': 'Proximity', 'K_Feasibility(3)': 'Feasibility', 'DiscriminativePower(9)': 'DiscriminativePower'}, inplace=True)
        
        
        metrics = ['Proximity', 'Feasibility', 'DiscriminativePower']
        directions = ['min', 'min', 'max']
        
        print(i_scores.columns)
        
        # Get counterfactual closest to ideal point
        iscores = i_scores[metrics].to_numpy()
        
        # Apply normalization in each feature
        iscores = (iscores - iscores.min(axis=0)) / (iscores.max(axis=0) - iscores.min(axis=0))
        
        pareto_mask = get_pareto_optimal_mask(iscores, directions).astype(bool)
        ideal_point = get_ideal_point(iscores, directions, pareto_mask)
        
        
        # Draw line to closest counterfactual from pareto front
        a = np.sum((iscores[pareto_mask] - iscores[pareto_mask].min(axis=0) / (iscores[pareto_mask].max(axis=0) - iscores[pareto_mask].min(axis=0))) - ideal_point, axis=1)
        idx = np.argmin(a**2) 
        coordinates_closest = iscores[pareto_mask][idx]
        
        print(pareto_mask.sum())
        print(len(pareto_mask))
        
        color1 = '#377eb8'
        color2 = '#ff7f00'
        color3 = '#e41a1c'
        
        
        metric1 = 0
        metric2 = 1
        # Plot 2D Pareto Front
        fig, ax = plt.subplots(figsize=(6, 5))
        plt.scatter(iscores[~pareto_mask, metric1], iscores[~pareto_mask, metric2], c=color1, alpha=0.8, label='Dominated', marker='s')
        plt.scatter(iscores[pareto_mask, metric1], iscores[pareto_mask, metric2], c=color2, alpha=0.8, label='Nondominated', marker='o')
        plt.scatter(ideal_point[metric1], ideal_point[metric2], c=color3, label='Ideal Point', marker='x')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Draw line to the best counterfactual
        ax.plot(
            [coordinates_closest[metric1], ideal_point[metric1]],
            [coordinates_closest[metric2], ideal_point[metric2]],
            color='#a65628',
            linestyle='dashed'
            )
        
        plt.xlabel(metrics[metric1])
        plt.ylabel(f'{metrics[metric2]}')
        # plot legend in the upper right corner 
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        plt.savefig(os.path.join(SAVE_PATH, f'{dataset}_pareto_front2D_{"-".join(metrics)}_{index}.eps'))
        plt.savefig(os.path.join(SAVE_PATH, f'{dataset}_pareto_front2D_{"-".join(metrics)}_{index}.png'))
        
        #plt.show()
        
        markersize = 45
        # Plot 3D Pareto Front
        fig3d = plt.figure(figsize=(6, 5))
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.scatter(iscores[~pareto_mask, 0], iscores[~pareto_mask, 1], iscores[~pareto_mask, 2], c=color1, alpha=1, label='Dominated', marker='s', s=markersize)
        ax3d.scatter(iscores[pareto_mask, 0], iscores[pareto_mask, 1], iscores[pareto_mask, 2], c=color2, alpha=0.8, label='Nondominated', marker='o', s=markersize)
        ax3d.scatter(ideal_point[0], ideal_point[1], ideal_point[2], c=color3, label='Ideal Point', marker='x', s=markersize+10)
        
        # Draw a line to the best counterfactual
        ax3d.plot(
            [coordinates_closest[0], ideal_point[0]],
            [coordinates_closest[1], ideal_point[1]],
            [coordinates_closest[2], ideal_point[2]],
            color='#a65628',
            linestyle='dashed'
            )
        
        # Change grid color to white
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        
        # Change view angle
        #ax3d.view_init(30, 135)
        
        # Set ticks
        # ax3d.set_xticks([0,1,2,3,4])
        # ax3d.set_yticks([0, 1, 2, 3])
        # ax3d.set_zticks([0.0,0.2,0.4,0.6,0.8,1.0])
        
        
        ax3d.set_xlabel(metrics[0])
        ax3d.set_ylabel(metrics[1])
        ax3d.set_zlabel(metrics[2])
        plt.tight_layout()

        plt.savefig(os.path.join(SAVE_PATH, f'{dataset}_pareto_front3D_{"-".join(metrics)}_{index}.eps'))
        plt.savefig(os.path.join(SAVE_PATH, f'{dataset}_pareto_front3D_{"-".join(metrics)}_{index}.png'))

        plt.show()

        break
