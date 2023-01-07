import numpy as np
import pandas as pd
from metrics.single_cf_metrics import *

def get_scores(counterfactuals, x, train_data, train_labels, mask) -> pd.DataFrame:
    x = x.to_numpy().flatten()
    train_data = train_data.to_numpy()

    # Get index in train_data of x
    x_index = np.argwhere(train_data==x)[0][0]
    x_label = train_labels[x_index]
    
    scores = []
    for cf in counterfactuals:
        cf = np.array(cf[0:85])
        prox = proximity(x, cf)
        f_changed = features_changed(x, cf)
        feasib = feasibility(x, cf, train_data) 
        actionab = actionability(x, cf, mask)
        preference_dcg = preference_dcg_score(x, cf, [0, 4, 1, 2, 3], k=5)
        discriminative_pow = discriminative_power(x, x_label, cf, '>50K', train_data, train_labels.to_numpy(), 30)

        # Create a dictionary with the scores
        scores.append({
            'cf': cf,
            'proximity': prox,
            'features_changed': f_changed,
            'feasibility': feasib,
            'actionability': actionab,
            'preference_dcg': preference_dcg,
            'discriminative_power': discriminative_pow,
        })

    
    df = pd.DataFrame(scores) 

    df['dispreference_dcg'] = df['preference_dcg'].max() - df['preference_dcg']
    df['non_discriminative_power'] = 1 - df['discriminative_power']

    return df



# Very slow for many datapoints.  Fastest for many costs, most readable
def is_pareto_efficient_dumb(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
    return is_efficient

def filter_non_valid(predict_fn, x, counterfactuals: pd.DataFrame):
    original_class_prob = predict_fn(x)
    valid_counterfactuals_indices = []
    for cf in counterfactuals.to_numpy():
        cf = cf.reshape(1, -1)
        cf_class_prob = predict_fn(cf)

        if np.argmax(cf_class_prob) != np.argmax(original_class_prob):
            valid_counterfactuals_indices.append(True)
        else:
            valid_counterfactuals_indices.append(False)

    return counterfactuals[valid_counterfactuals_indices]

def remove_duplicates(counterfactuals: pd.DataFrame):
    return counterfactuals.drop_duplicates().reset_index(drop=True)

def filter_non_actionable(counterfactuals: pd.DataFrame, query_instance: pd.DataFrame, non_actionable_features: list):
    # filter if the counterfactual is not actionable
    # counterfactual is not actionable if it has any non-actionable features changed in respect to the query instance
    # non-actionable features are features that are not allowed to be changed in the counterfactual

    print(counterfactuals[non_actionable_features].shape)
    print(query_instance[non_actionable_features].shape)
    
    true_mask = counterfactuals[non_actionable_features].to_numpy()[:] == query_instance[non_actionable_features].to_numpy()

    true_indices = []
    for i, true in enumerate(true_mask.all(axis=1)):
        if true: 
            true_indices.append(i)
   
    return counterfactuals.take(true_indices)
   
    