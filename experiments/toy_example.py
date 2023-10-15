import pandas as pd
import numpy as np
import os
from experiment_utils import load_data, get_pareto_optimal_mask, get_ideal_point

PATH = os.path.join(os.getcwd(), 'experiments', '2023-03-12', 'counterfactuals', 'adult_tensorflow_i48_2023-03-12.csv')
PATH_SCORES = os.path.join(os.getcwd(), 'experiments', '2023-03-12', 'scores', 'adult_tensorflow_i48_2023-03-12.csv')
# experiments/2023-03-12/counterfactuals/adult_tensorflow_i48_2023-03-12.csv

original_x = [24,10,0,0,30,"Self-emp-not-inc","Never-married","Prof-specialty","Asian-Pac-Islander","Male","United-States", 0]

raw_data = pd.read_csv(PATH)
data = raw_data.copy(deep=True)
orginal = pd.DataFrame([original_x], columns=data.drop(columns=['explainer']).columns)

print(f'All data: {len(data)}')
print()

invalid = data[data["income"] == orginal["income"].iloc[0]]
print(f'Invalid data: {len(invalid)}. Left: {len(data) - len(invalid)}')
print(invalid)
print()

data = data.drop(invalid.index)

nonactionable = data[
    (data['race'] != orginal["race"].iloc[0]) + \
    (data['sex'] != orginal["sex"].iloc[0]) + \
    (data['native.country'] != orginal["native.country"].iloc[0])
    ]

print(f'Nonactionable {len(nonactionable)}. Left: {len(data) - len(nonactionable)}')
print(nonactionable)



data = data.drop(nonactionable.index)


scores = pd.read_csv(PATH_SCORES).drop(invalid.index).drop(nonactionable.index)


index = 48
i_scores = scores.drop(columns=['Feasibility', 'DCG@6', 'explainer'])

i_scores.rename(columns={'Proximity': 'Proximity', 'K_Feasibility(3)': 'Feasibility', 'DiscriminativePower(9)': 'DiscriminativePower'}, inplace=True)


metrics = ['Proximity', 'Feasibility', 'DiscriminativePower']
directions = ['min', 'min', 'max']


# Get counterfactual closest to ideal point
iscores = i_scores[metrics].to_numpy()

# Apply normalization in each feature
# iscores = (iscores - iscores.min(axis=0)) / (iscores.max(axis=0) - iscores.min(axis=0))

pareto_mask = get_pareto_optimal_mask(iscores, directions).astype(bool)
ideal_point = get_ideal_point(iscores, directions, pareto_mask)
print(f'Ideal point coordinates: {ideal_point}')


print(f'Applying dominance relation drops: {len(data) - pareto_mask.sum()}. Left on Pareto front: {pareto_mask.sum()}')



a = np.sum((iscores[pareto_mask] - iscores[pareto_mask].min(axis=0) / (iscores[pareto_mask].max(axis=0) - iscores[pareto_mask].min(axis=0))) - ideal_point, axis=1)
idx = np.argmin(a**2) 
print('best:', idx)
coordinates_closest = iscores[pareto_mask][idx]

print(data.iloc[4])
print(pareto_mask)



print(f'Invalid indices: {invalid.index.tolist()}')
print(f'Nonactionable indices: {nonactionable.index.tolist()}')
print(f'Pareto Front indices: {data[pareto_mask].index.tolist()}')
print(f'The rest indices: {data[~pareto_mask].index.tolist()}')