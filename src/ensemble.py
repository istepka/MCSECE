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
        self.features_order_before_split = constraints_config_dictionary['features_order_nonsplit']
        self.dataset_shortname = constraints_config_dictionary['dataset_shortname']
        self.map_target_feature_to_encoded = constraints_config_dictionary['map_target_to_encoded']
        

        self.train_dataset_pd[self.features_order_before_split]

        # All non ohe features in proper order, without target feature name
        self.all_features_without_target = train_dataset.columns.tolist()
        self.all_features_without_target.remove(self.feature_name_to_predict)
        self.features_order_before_split_without_target = self.features_order_before_split.copy()
        self.features_order_before_split_without_target.remove(self.feature_name_to_predict)

        

        # Map of feature names before ohe to feature names in ohe encoding
        self.categorical_features_map_to_thier_splits = constraints_config_dictionary['categorical_features_map_to_thier_splits']

        # Transform train data
        self.train_dataset_ohe_normalized = self.transform_to_normalized_ohe(self.train_dataset_pd)

        # Create mask for actionable features in one-hot-encoded form. 1 indicates actionable features
        self.actionable_mask_ohe = np.array([ 
            1 if any([act in x for act in self.actionabe_columns_names]) else 0 
            for x in self.features_order_after_split], 
            dtype='bool'
            )
        self.actionable_mask_ohe_indices = np.where(self.actionable_mask_ohe)[0].tolist()

        # Models definitions and init
        # dice
        self.dice_model: DiceModel
        self.__init_dice()

        # cfec
        self.cfec_model: CfecEceModel
        self.__init_cfec_ece()
        self.__fit_cfec_ece()



        # Init empty properties for linter
        self.all_counterfactuals: pd.DataFrame
        self.valid_counterfactuals: pd.DataFrame
        self.valid_actionable_counterfactuals: pd.DataFrame

        


    def generate_counterfactuals(self, query_instance: pd.DataFrame) -> pd.DataFrame:
        '''
        Can ask only for one query instance at a time.
        Has to be in pd.DataFrame format.
        '''
        assert query_instance.shape[0] == 1, 'Only one query instance at a time supported!'

        counterfactuals = pd.DataFrame(columns=self.all_features_without_target + ['explainer'])

        query_instance = query_instance[self.features_order_before_split_without_target]

        query_instance_norm_ohe = self.transform_to_normalized_ohe(query_instance)

        query_instance_class_int = int(np.argmax(self.model_to_explain.predict(query_instance_norm_ohe.to_numpy().astype("float64"))))
        print(f'X class: {query_instance_class_int}')

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
        tmp = cfec_cfs_norm_ohe.to_numpy()
        if cfec_cfs_norm_ohe is not None and len(cfec_cfs_norm_ohe) > 0:
            cfec_cfs = self.transform_from_norm_ohe(cfec_cfs_norm_ohe)
            cfec_cfs['explainer'] = cfec_explainers
        else:
            cfec_cfs = counterfactuals.copy()
        print(f'CFEC generated: {cfec_cfs.shape[0]}')

        counterfactuals = pd.concat([counterfactuals, dice_cfs, cfec_cfs], ignore_index=True)
        #counterfactuals = self.__clip_to_ranges(counterfactuals)
        counterfactuals[self.feature_name_to_predict] = self.__get_prediction_class_to_counterfactuals(counterfactuals)

        self.all_counterfactuals = counterfactuals.copy()
        self.valid_counterfactuals = self.__filter_only_valid(counterfactuals, query_instance)
        self.valid_actionable_counterfactuals = self.__filter_non_actionable(self.valid_counterfactuals, query_instance)

        return counterfactuals


    
    def __init_dice(self) -> None:
        # Get Dice-format backend name
        if self.model_backend_name == 'sklearn':
            backend = 'sklearn'
        else:
            backend = 'TF2'

        # encode target feature
        train_df = self.train_dataset_pd.copy()

        for value, encoding in self.map_target_feature_to_encoded.items():
            train_df[self.feature_name_to_predict] = train_df[self.feature_name_to_predict].replace(value, encoding)
        
        #print(train_df[self.feature_name_to_predict])

        self.dice_model = DiceModel(
            train_dataset=train_df,
            continuous_features=self.continuous_columns_names,
            categorical_features=self.categorical_columns_names,
            target=self.feature_name_to_predict,
            backend=backend,
            model=self.model_to_explain
        )
    
    def __dice_generate_counterfactuals(self, query_instance: pd.DataFrame, cfs_total: int = 10, desired_class: str | int = 'opposite') -> pd.DataFrame:
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
            columns_to_change=self.actionable_mask_ohe_indices,
            cadex_max_feature_changes=15,
            cadex_max_epochs=20
        )

    def __fit_cfec_ece(self, fimap_load_models_date: str = '2023-01-26') -> None:
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

        return query_instance_ohe_norm[self.features_order_after_split] # Make sure that correct order is mantaineed

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
            
    def __clip_to_ranges(self, counterfactuals: pd.DataFrame) -> pd.DataFrame:
        '''Clip values outside of the defined ranges'''
        for feature, (lower, upper) in self.feature_ranges.items():
            #print(ranges)
           # lower, upper = ranges[0], ranges[1]
           counterfactuals.loc[counterfactuals[feature] < lower, feature] = lower
           counterfactuals.loc[counterfactuals[feature] > upper, feature] = upper
        return counterfactuals

    def __get_prediction_class_to_counterfactuals(self, counterfactuals: pd.DataFrame) -> pd.DataFrame:
        '''
        Assign prediction class to counterfactuals.
        '''
        cfs_norm_ohe = self.transform_to_normalized_ohe(counterfactuals).to_numpy().astype('float64')

        preds = np.argmax(self.model_to_explain.predict(cfs_norm_ohe), axis=1)

        return pd.DataFrame(preds, columns=[self.feature_name_to_predict])

    def __filter_only_valid(self, counterfactuals: pd.DataFrame, query_instance: pd.DataFrame) -> pd.DataFrame:
        '''
        Filter counterfactuals to retrieve only valid ones.
        '''
        assert self.feature_name_to_predict in counterfactuals.columns, 'Target feature should be in counterfactuals DataFrame'
        assert all(pd.notna(counterfactuals[self.feature_name_to_predict])), 'All counterfactuals should have their predicted class assigned'


        query_instance_ohe_norm = self.transform_to_normalized_ohe(query_instance).to_numpy().astype('float64')
        query_class = np.argmax(self.model_to_explain.predict(query_instance_ohe_norm), axis=1)

        mask = counterfactuals[self.feature_name_to_predict].astype('int') != int(query_class)

        return counterfactuals[mask]

    def __filter_non_actionable(self, counterfactuals: pd.DataFrame, query_instance: pd.DataFrame) -> pd.DataFrame:
        '''
        Filter counterfactuals that are non-actionable (changes on features that were Freezed)
        '''
        mask = np.ones(counterfactuals.shape[0]).astype('bool')

        for col in self.non_actionable_columns_names:
            if col in self.categorical_columns_names:
                _mask = counterfactuals[col].to_numpy().astype('str') == query_instance[col].to_numpy().astype('str')
            else: # continous
                _mask = np.isclose(counterfactuals[col].to_numpy().astype('float64'), query_instance[col].to_numpy().astype('float64'))
            mask = mask & _mask

        return counterfactuals[mask]

    def get_all_counterfactuals(self) -> pd.DataFrame:
        '''Get all counterfactuals that were found. No guarantees of validity and actionability.'''
        return self.all_counterfactuals

    def get_valid_counterfactuals(self) -> pd.DataFrame:
        '''Get all counterfactuals that were found and are guaranted to alter the prediction class.'''
        return self.valid_counterfactuals

    def get_valid_and_actionable_counterfactuals(self) -> pd.DataFrame:
        '''
        Get all counterfactuals that were found and are guaranted to alter 
        the prediction class and to be actionable according to constraints given.
        '''
        return self.valid_actionable_counterfactuals

     

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
    instance_to_explain_index = 8908

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

    print('----'*10)
    print('ALL', enseble.get_all_counterfactuals())
    print('----'*10)
    print('VALID', enseble.get_valid_counterfactuals())
    print('----'*10)
    print('VALID ACT', enseble.get_valid_and_actionable_counterfactuals())
    print('----'*10)


    # print(f'Query instance: {train_dataset["income"].iloc[instance_to_explain_index]}')

    