import pandas as pd
import numpy as np
import json
from typing import List, Dict, Tuple
from collections import defaultdict
from sklearn.utils import shuffle

def imbalance_aware_split(df: pd.DataFrame, target_feature_col: str, test_size_per_class: int = 200) -> Tuple[pd.DataFrame, pd.DataFrame]:
    shuffled_df = shuffle(df.copy(), random_state=44)

    train = pd.DataFrame(columns=shuffled_df.columns)
    test = pd.DataFrame(columns=shuffled_df.columns)

    classes = np.unique(shuffled_df[target_feature_col])
    for cl in classes:
        class_only = shuffled_df[shuffled_df[target_feature_col] == cl] # only from class
         
        test = pd.concat([test, class_only.iloc[0:test_size_per_class]], ignore_index=True)
        train = pd.concat([train, class_only.iloc[test_size_per_class:]], ignore_index=True)

    test = shuffle(test, random_state=44).reset_index(drop=True)
    train = shuffle(train, random_state=44).reset_index(drop=True)

    return train, test 



def generate_dataset_files_and_config(
    dataset_name: str, data: pd.DataFrame, categorical_columns: List[str],
    continuous_columns: List[str], target_column: str,
    monotonic_increasing_columns: List[str], monotonic_decreasing_columns: List[str],
    freeze_columns: List[str], feature_ranges: Dict[str, List[int]],
    test_size_per_class: int
    ) -> None:
    '''
    Generate config file (json) for dataset and separate train/test (normal and ohe) datasets.

    Params: 
    `dataset_name`: shortname for dataset e.g. 'adult'  
    `data`: dataframe containing raw (non-split) data  
    `categorical_columns`: list of names of categorical columns  
    `continuous_columns`: list of names of continuous columns  
    `target_column`: string with the name of target column  
    `monotonic_increasing_columns`: list of names of columns that are allowed only to increase   
    `monotonic_decreasing_columns`: list of names of columns that are allowed only to decrease   
    `freeze_columns`: list of columns that are non-actionable and should not change in counterfactual  
    `feature_ranges`: dictionary of features and their specified ranges. e.g {'age': [18, 75]}     
    `test_size_per_class`: number of instances that each class should have in the test set
    '''
    columns_order_nonsplit = continuous_columns + categorical_columns + [target_column]

    data_to_concat = []
    data_to_concat.append(data[target_column])

    for cont_column in continuous_columns:
        data_to_concat.append(data[cont_column])

    # Split categorical columns into one-hot-encoded
    categorical_columns_after_split = []
    categorical_features_map_to_thier_splits = {}
    constraints_on_features = defaultdict(lambda: list())

    for column_to_split in categorical_columns:
        dummies = pd.get_dummies(data[column_to_split], prefix=column_to_split)
        categorical_columns_after_split += dummies.columns.to_list()
        data_to_concat.append(dummies)
        categorical_features_map_to_thier_splits[column_to_split] = dummies.columns.to_list()

        # Add constraints for columns
        for _column in dummies.columns:
            constraints_on_features[_column].append('onehot')

            if column_to_split in freeze_columns:
                constraints_on_features[_column].append('non-actionable')

    final_df = pd.concat(data_to_concat, axis=1)

    columns_order_after_split_without_target = final_df.drop([target_column], axis=1).columns.to_list()
    print(final_df.columns)

    # Get first occurence indices and ohe lenghts
    feature_counts = defaultdict(lambda: 0)
    feature_first_occurrence = {}

    for idx, column in enumerate(final_df[columns_order_after_split_without_target].columns.tolist()):
        for col in continuous_columns + categorical_columns:
            if col in column:
                # Count how many ohe values for feature
                feature_counts[col] += 1

                # Get first occurence index 
                if col not in feature_first_occurrence.keys():
                    feature_first_occurrence[col] = idx 

    encoding_target = {col: i for i, col in enumerate(pd.get_dummies(data[target_column]).columns)}

    features_monotonicity_dict = {}
    for feature in monotonic_increasing_columns:
        features_monotonicity_dict[feature] = 'increasing'
    for feature in monotonic_decreasing_columns:
        features_monotonicity_dict[feature] = 'decreasing'

    config = dict()
    config['dataset_shortname'] = dataset_name
    config['features_order_nonsplit'] = columns_order_nonsplit
    config['categorical_features_nonsplit'] = categorical_columns
    config['continuous_features_nonsplit'] = continuous_columns
    config['target_feature'] = target_column
    config['features_count_split_without_target'] = len(columns_order_after_split_without_target)
    config['non_actionable_features'] = freeze_columns
    config['actionable_features'] = list(set(categorical_columns + continuous_columns) - set(freeze_columns))
    config['features_count_nonsplit'] = feature_counts
    config['feature_first_occurrence_after_split'] = feature_first_occurrence
    config['categorical_features_map_to_thier_splits'] = categorical_features_map_to_thier_splits
    config['features_monotonocity'] = features_monotonicity_dict
    config['feature_ranges'] = feature_ranges
    config['map_target_to_encoded'] = encoding_target
    config['features_order_after_split'] = columns_order_after_split_without_target

    # Save config file
    with open(f'data/{dataset_name}_constraints.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Save one-hot-encoded dataset in correct order and target feature at the end
    cleaned_df = final_df[columns_order_after_split_without_target + [target_column]]
    cleaned_df.to_csv(f'data/{dataset_name}_ohe.csv', index=False)


    # Create Train Test split
    train_df, test_df = imbalance_aware_split(data[columns_order_nonsplit], target_column, test_size_per_class=test_size_per_class)
    train_ohe_df, test_ohe_df = imbalance_aware_split(cleaned_df, target_column, test_size_per_class=test_size_per_class)


    train_df.to_csv(f'data/{dataset_name}_train.csv', index=False)
    test_df.to_csv(f'data/{dataset_name}_test.csv', index=False)
    train_ohe_df.to_csv(f'data/{dataset_name}_train_ohe.csv', index=False)
    test_ohe_df.to_csv(f'data/{dataset_name}_test_ohe.csv', index=False)


    print(train_df.info())
    print(test_df.info())

    print(train_ohe_df.info())
    print(test_ohe_df.info())


def german():
    # GERMAN
    raw_df = pd.read_csv('data/german.csv')
    categorical_columns = [
        'checking_status', 'credit_history', 'purpose', 
        'savings_status', 'employment', 'personal_status', 
        'other_parties', 'property_magnitude', 'other_payment_plans', 
        'housing', 'job', 'own_telephone', 'foreign_worker'
        ]
    continuous_columns = [
        'duration', 'credit_amount', 'installment_commitment', 
        'residence_since', 'age', 'existing_credits', 'num_dependents'
        ]
    target_column = 'class'

    monotonic_increase_columns = []
    monotonic_decrease_columns = []
    freeze_columns = ['foreign_worker']
    feature_ranges = {
        'duration': [int(raw_df['duration'].min()), int(raw_df['duration'].max())],
        'credit_amount': [int(raw_df['credit_amount'].min()), int(raw_df['credit_amount'].max())],
        'installment_commitment': [int(raw_df['installment_commitment'].min()), int(raw_df['installment_commitment'].max())],
        'residence_since': [int(raw_df['residence_since'].min()), int(raw_df['residence_since'].max())],
        'existing_credits': [int(raw_df['existing_credits'].min()), int(raw_df['existing_credits'].max())],
        'num_dependents': [int(raw_df['num_dependents'].min()), int(raw_df['num_dependents'].max())],
        'age': [18, int(raw_df['age'].max())],
    }


    generate_dataset_files_and_config(
        dataset_name='german',
        data=raw_df,
        categorical_columns=categorical_columns,
        continuous_columns=continuous_columns,
        target_column=target_column,
        monotonic_increasing_columns=monotonic_increase_columns,
        monotonic_decreasing_columns=monotonic_decrease_columns,
        freeze_columns=freeze_columns,
        feature_ranges=feature_ranges,
        test_size_per_class=50
    )

    print(raw_df[target_column].value_counts())

def fico():
    
    raw_df = pd.read_csv('data/fico.csv')
    categorical_columns = []
    continuous_columns = raw_df.columns.tolist()
    continuous_columns.remove('RiskPerformance')
    target_column = 'RiskPerformance'
    
    freeze_columns = ['ExternalRiskEstimate']
    feature_ranges = {
        'PercentTradesNeverDelq': [0, 100],
        'PercentInstallTrades': [0, 100],
        'PercentTradesWBalance': [0, 100],
    }
    monotonic_increase_columns = []
    monotonic_decrease_columns = []
    
    # In fico dataset negative values mean that the value is missing
    mask_negative = ~np.any(raw_df[continuous_columns] < 0, axis=1)
    raw_df = raw_df[mask_negative]
    
    generate_dataset_files_and_config(
        dataset_name='fico',
        data=raw_df,
        categorical_columns=categorical_columns,
        continuous_columns=continuous_columns,
        target_column=target_column,
        monotonic_increasing_columns=monotonic_increase_columns,
        monotonic_decreasing_columns=monotonic_decrease_columns,
        freeze_columns=freeze_columns,
        feature_ranges=feature_ranges,
        test_size_per_class=125 # ~2500 instances -> 10% test size -> 5% per class = 125
    )

def compas():
    raw_df = pd.read_csv('https://github.com/propublica/compas-analysis/raw/master/compas-scores-two-years.csv')
    raw_df = raw_df[raw_df['type_of_assessment'] == 'Risk of Recidivism']

    features = [
        'sex', 'age', 'race', 'juv_fel_count', 
        'decile_score', 'juv_misd_count', 'juv_other_count', 
        'priors_count', 'c_days_from_compas', 'c_charge_degree',
        'two_year_recid'
        ]

    categorical_columns = ['sex', 'race', 'c_charge_degree']
    continuous_columns = ['age', 'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_days_from_compas']
    target_column = 'two_year_recid'
    freeze_columns = ['age', 'sex', 'race', 'c_charge_degree']

    feature_ranges = {
        'age': [18, 100],
        'decile_score': [0, 10],
    }

    raw_df = raw_df[features]
    raw_df = raw_df.dropna(how='any', axis=0)
    
    generate_dataset_files_and_config(
        dataset_name='compas',
        data=raw_df,
        categorical_columns=categorical_columns,
        continuous_columns=continuous_columns,
        target_column=target_column,
        monotonic_increasing_columns=[],
        monotonic_decreasing_columns=[],
        freeze_columns=freeze_columns,
        feature_ranges=feature_ranges,
        test_size_per_class=125 # ~7200 instances -> 10% test size -> 5% per class = 125
    )


if __name__ == '__main__':
    
    #german()
    #fico()
    compas()
