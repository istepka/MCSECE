import pandas as pd 
import numpy as np

for dataset in ['adult', 'compas', 'german', 'fico']:

    df = pd.read_csv(f'experiments/tmp_results2/experiment1_{dataset}.csv_valid.csv', index_col=0)
    
    index_to_remove = ['dice', 'wachter', 'cfproto']
    df = df.drop(index_to_remove, axis=0)
    df.rename(index={'dice-1': 'Dice', 'wachter-1': 'Wachter', 'cfproto-1': 'CFProto'}, inplace=True)

    ascending=['discriminative_power_9', 'coverage', 'actionable']
    part_1 = df[ascending].rank(axis=0, method='min', ascending=False)

    part_2 = df.drop(ascending, axis=1).rank(axis=1, method='min', ascending=True)

    df['rank'] = (part_1.sum(axis=1) + part_2.sum(axis=1)) / len(df.columns)
    df['rank'] = df['rank'].round(2)

    df.to_latex(f'experiments/tmp_results2/experiment1_{dataset}_valid_rank.tex')