# Libs
from typing import Dict, List
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier

# Own
from dice import DiceModel 
from cfec_ece import CfecEceModel

class Ensemble:

    def __init__(self, train_dataset: pd.DataFrame, constraints_config_dictionary: Dict,
        model_to_explain: tf.keras.Model | RandomForestClassifier,
        list_of_explainers: List[str] = ['dice', 'fimap', 'cadex', 'wachter', 'cem'],
    ) -> None:
        
        self.train_dataset_pd = train_dataset
        self.constraints = constraints_config_dictionary
        self.model_to_explain = model_to_explain
        self.list_of_explainers = list_of_explainers

        if isinstance(model_to_explain, tf.keras.Model):
            self.model_backend_name = 'tensorflow' 
        else:
            self.model_backend_name = 'sklearn'

        # Column names on not-transformed data 
        self.categorical_columns_names = constraints_config_dictionary['categorical_features_nonsplit']
        self.continuous_columns_names = constraints_config_dictionary['continuous_features_nonsplit']
        self.non_actionable_columns_names = constraints_config_dictionary['non_actionable_features']
        self.actionabe_columns_names = constraints_config_dictionary['actionable_features']
        self.feature_ranges = constraints_config_dictionary['feature_ranges']
        self.feature_value_counts = constraints_config_dictionary['features_count_nonsplit']
        self.feature_first_occurrence_index_after_split = constraints_config_dictionary['feature_first_occurrence_after_split']
        self.features_monotonicity = constraints_config_dictionary['features_monotonocity']
        self.feature_name_to_predict = constraints_config_dictionary['target_feature']
        self.features_order_after_split = constraints_config_dictionary['features_order_after_split']
        self.dataset_shortname = constraints_config_dictionary['dataset_shortname']

        # All non ohe features in proper order, without target feature name
        self.all_features_without_target = train_dataset.columns.tolist()
        self.all_features_without_target.remove(self.feature_name_to_predict)
        

        # Map of feature names before ohe to feature names in ohe encoding
        self.categorical_features_map_to_thier_splits = constraints_config_dictionary['categorical_features_map_to_thier_splits']

        # Transform train data
        self.train_dataset_ohe_normalized = self.transform_to_normalized_ohe(self.train_dataset_pd)

        # Create mask for actionable features in one-hot-encoded form. 1 indicates actionable features
        self.actionable_mask_indices_ohe = [
            1 if any([act in x for act in self.actionabe_columns_names]) else 0 
            for x in self.features_order_after_split
            ]

        # Models definitions and init
        # dice
        self.dice_model: DiceModel
        self.__init_dice()
        # cfec
        self.cfec_model: CfecEceModel
        self.__init_cfec_ece()
        self.__fit_cfec_ece()

        


    def generate_counterfactuals(self, query_instance: pd.DataFrame) -> pd.DataFrame:
        '''
        Can ask only for one query instance at a time.
        Has to be in pd.DataFrame format.
        '''
        assert query_instance.shape[0] == 1, 'Only one query instance at a time supported!'

        counterfactuals = pd.DataFrame(columns=self.all_features_without_target + ['explainer'])
        query_instance_norm_ohe = self.transform_to_normalized_ohe(query_instance)

        # DICE 
        dice_cfs = self.__dice_generate_counterfactuals(query_instance=query_instance)
        if dice_cfs is not None:
            dice_cfs['explainer'] = 'dice'
        else:
            dice_cfs = counterfactuals.copy()
        print(f'Dice generated: {dice_cfs.shape[0]}')


        # CFEC
        cfec_cfs_norm_ohe, cfec_explainers = self.cfec_model.generate_counterfactuals(
            query_instance=query_instance_norm_ohe.iloc[0]
        )
        if cfec_cfs_norm_ohe is not None and len(cfec_cfs_norm_ohe) > 0:
            cfec_cfs = self.transform_from_norm_ohe(cfec_cfs_norm_ohe)
            cfec_cfs['explainer'] = cfec_explainers
        else:
            cfec_cfs = counterfactuals.copy()
        print(f'CFEC generated: {cfec_cfs.shape[0]}')

        counterfactuals = pd.concat([counterfactuals, dice_cfs, cfec_cfs], ignore_index=True)
        return counterfactuals


    
    def __init_dice(self) -> None:
        # Get Dice-format backend name
        if self.model_backend_name == 'sklearn':
            backend = 'sklearn'
        else:
            backend = 'TF2'

        self.dice_model = DiceModel(
            train_dataset=self.train_dataset_pd,
            continuous_features=self.continuous_columns_names,
            categorical_features=self.categorical_columns_names,
            target=self.feature_name_to_predict,
            backend=backend,
            model=self.model_to_explain
        )
    
    def __dice_generate_counterfactuals(self, query_instance: pd.DataFrame, cfs_total: int = 20, desired_class: str = 'opposite') -> pd.DataFrame:
        dice_counterfactuals_df = self.dice_model.generate_counterfactuals(
            query_instance=query_instance,
            total_CFs=cfs_total,
            desired_class=desired_class,
            features_to_vary=self.actionabe_columns_names,
            permitted_range=self.feature_ranges,
        )

        return dice_counterfactuals_df

    def __init_cfec_ece(self) -> None:
        self.cfec_model = CfecEceModel(
            train_data_normalized=self.train_dataset_ohe_normalized,
            constraints_dictionary=self.constraints,
            model = self.model_to_explain,
            columns_to_change=self.actionable_mask_indices_ohe,
            cadex_max_feature_changes=15,
            cadex_max_epochs=20
        )

    def __fit_cfec_ece(self, fimap_load_models_date: str = '2023-01-17') -> None:
        self.cfec_model.fit(
            fimap_load_string=f'{self.dataset_shortname}_{self.model_backend_name}|{fimap_load_models_date}'
            )


    def transform_to_normalized_ohe(self, query_instance: pd.DataFrame) -> pd.DataFrame:
        '''Transform original dataframe into one-hot-encoded and normalized form.'''
        query_instance_ohe = transform_to_sparse(
            _df = query_instance,
            original_df=self.train_dataset_pd.drop(columns="income"),
            categorical_features=self.categorical_columns_names,
            continuous_features=self.continuous_columns_names
        )

        query_instance_ohe_norm = min_max_normalization(
            _df=query_instance_ohe,
            original_df=self.train_dataset_pd.drop(columns="income"),
            continuous_features=self.continuous_columns_names
        )

        return query_instance_ohe_norm

    def transform_from_norm_ohe(self, query_instance_norm_ohe: pd.DataFrame) -> pd.DataFrame:
        '''Transform from one-hot-encoded normalized form into original dataframe.'''
        query_instance_ohe = inverse_min_max_normalization(
            _df=query_instance_norm_ohe,
            original_df=self.train_dataset_pd.drop(columns="income"),
            continuous_features=self.continuous_columns_names
        )

        query_instance = inverse_transform_to_sparse(
            sparse_df=query_instance_ohe,
            original_df=self.train_dataset_pd.drop(columns="income"),
            categorical_features=self.categorical_columns_names,
            continuous_features=self.continuous_columns_names
        )

        return query_instance
            


if __name__ == '__main__':
    explained_model_backend = 'tensorflow' # 'sklearn' or 'tensorflow'


    from sklearn.model_selection import train_test_split
    import json
    from utils.transformations import min_max_normalization, inverse_min_max_normalization, transform_to_sparse, inverse_transform_to_sparse
    import warnings
    import pickle

    warnings.filterwarnings('ignore', category=UserWarning) #Ignore sklearn "RF fitted with FeatureNames"

    train_dataset = pd.read_csv("data/adult.csv")
    dataset_name = 'adult'
    instance_to_explain_index = 890

    with open('data/adult_constraints.json', 'r') as f:
        constr = json.load(f)

    if explained_model_backend == 'sklearn':
        # SKLEARN
        with open('models/adult_RF.pkl', 'rb') as f:
            explained_model = pickle.load(f)
    else: 
        # TENSORFLOW
        explained_model = tf.keras.models.load_model('models/adult_NN/')


    train_dataset = train_dataset[constr['features_order_nonsplit']]
    query_instance = train_dataset.drop(columns="income")[instance_to_explain_index:instance_to_explain_index+1]


    enseble = Ensemble(
        train_dataset=train_dataset,
        constraints_config_dictionary=constr,
        model_to_explain=explained_model
    )

    cfs = enseble.generate_counterfactuals(query_instance)

    print(cfs)