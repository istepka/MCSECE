# Libs
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from time import time
import json 
import warnings
import logging

# Own
from explainers.dice import DiceModel 
from explainers.cfec_ece import CfecEceModel
from utils.transform import Transformer

# Hyperparameter for methods which we set to generate N explanations
DESIRED_CF_COUNTS = {
    'wachter': 10,
    'dice': 20
}

class Ensemble:

    def __init__(self, train_dataset: pd.DataFrame, constraints_config_dictionary: Dict,
        model_to_explain: tf.keras.Model | RandomForestClassifier, model_path: str,
        list_of_explainers: List[str] = ['dice', 'cfec', 'wachter', 'cem', 'cfproto', 'carla'],
    ) -> None:
        
        self.train_dataset_pd = train_dataset
        self.constraints = constraints_config_dictionary
        self.model_to_explain = model_to_explain
        self.model_path = model_path
        self.list_of_explainers = list_of_explainers

        if isinstance(model_to_explain, tf.keras.Model):
            self.model_backend_name = 'tensorflow' 
        else:
            self.model_backend_name = 'sklearn'
            
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
        
        # Initialize transformer for data transformations
        self.transformer = Transformer(
            train_dataset=self.train_dataset_pd,
            categorical_columns_names=self.categorical_columns_names,
            continuous_columns_names=self.continuous_columns_names,
            feature_name_to_predict=self.feature_name_to_predict,
            features_order_after_split=self.features_order_after_split,
        )

        # Transform train data
        self.train_dataset_ohe_normalized = self.transformer.transform_to_normalized_ohe(self.train_dataset_pd)

        # Create mask for actionable features in one-hot-encoded form. 1 indicates actionable features
        self.actionable_mask_ohe = np.array([ 
            1 if any([act in x for act in self.actionabe_columns_names]) else 0 
            for x in self.features_order_after_split], 
            dtype='bool'
            )
        self.actionable_mask_ohe_indices = np.where(self.actionable_mask_ohe)[0].tolist()
        
        # Models definitions and init
        if 'dice' in self.list_of_explainers:
            # dice
            self.dice_model: DiceModel
            self.__init_dice()

        # cfec
        if 'cfec' in self.list_of_explainers:
            self.cfec_model: CfecEceModel
            self.__init_cfec_ece()
            self.__fit_cfec_ece()

        # Init empty properties for linter
        self.all_counterfactuals: pd.DataFrame
        self.valid_counterfactuals: pd.DataFrame
        self.valid_actionable_counterfactuals: pd.DataFrame
        self.exectution_times: Dict[str, int] = dict() 
        

    def generate_counterfactuals(self, query_instance: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''
        Can ask only for one query instance at a time.
        Has to be in pd.DataFrame format.

        Return: valid and actionable countefactuals as data frame
        '''
        assert query_instance.shape[0] == 1, 'Only one query instance at a time supported!'

        counterfactuals = pd.DataFrame(columns=self.all_features_without_target + ['explainer'])

        query_instance = query_instance[self.features_order_before_split_without_target]

        query_instance_norm_ohe = self.transformer.transform_to_normalized_ohe(query_instance)

        query_instance_class_int = int(np.argmax(self.model_to_explain.predict(query_instance_norm_ohe.to_numpy().astype("float64"))))
        desired_class = 1 - query_instance_class_int
        logging.debug(f'X class: {query_instance_class_int}')
        logging.debug(f'Desired cf class: {desired_class}')

        to_concat = list()

        if 'dice' in self.list_of_explainers:
            # generate DICE 
            timer = time()
            dice_cfs = self.__dice_generate_counterfactuals(query_instance=query_instance)
            if dice_cfs is not None:
                dice_cfs['explainer'] = 'dice'
                to_concat.append(dice_cfs)
            else:
                dice_cfs = counterfactuals.copy()
            self.exectution_times['dice'] = time() - timer
            logging.debug(f'Dice generated: {dice_cfs.shape[0]}')


        if 'cfec' in self.list_of_explainers:
            # generate CFEC
            timer = time()
            cfec_cfs_norm_ohe, cfec_explainers = self.cfec_model.generate_counterfactuals(
                query_instance=query_instance_norm_ohe.iloc[0]
            )
            if cfec_cfs_norm_ohe is not None and len(cfec_cfs_norm_ohe) > 0:
                cfec_cfs = self.transformer.transform_from_norm_ohe(cfec_cfs_norm_ohe)
                cfec_cfs['explainer'] = cfec_explainers
                to_concat.append(cfec_cfs)
            else:
                cfec_cfs = counterfactuals.copy()
            self.exectution_times['cfec'] = time() - timer
            logging.debug(f'CFEC generated: {cfec_cfs.shape[0]}')


        if 'wachter' in self.list_of_explainers:
            # generate Wachter
            timer = time()
            wachter_cfs = self.__wachter_generate_counterfactuals(
                query_instance=query_instance, 
                desired_class=desired_class, 
                total_cfs=DESIRED_CF_COUNTS['wachter'])
            
            if wachter_cfs is None or len(wachter_cfs) == 0:
                wachter_cfs = counterfactuals.copy()
                
            wachter_cfs['explainer'] = 'wachter'
            to_concat.append(wachter_cfs)
            self.exectution_times['wachter'] = time() - timer
            logging.debug(f'Wachter generated: {wachter_cfs.shape[0]}')
            
        if 'cem' in self.list_of_explainers:
            # generate CEM
            timer = time()
            cem_cfs = self.__cem_generate_counterfactuals(query_instance)
            if cem_cfs is None or len(cem_cfs) == 0:
                cem_cfs = counterfactuals.copy()
            cem_cfs['explainer'] = 'cem'
            to_concat.append(cem_cfs)
            self.exectution_times['cem'] = time() - timer
            logging.debug(f'CEM generated: {cem_cfs.shape[0]}')
            
        if 'cfproto' in self.list_of_explainers:
            # generate CfProto
            timer = time()
            cfproto_cfs = self.__cfproto_generate_counterfactuals(query_instance)
            if cfproto_cfs is None or len(cfproto_cfs) == 0:
                cfproto_cfs = counterfactuals.copy()
            cfproto_cfs['explainer'] = 'cfproto'
            to_concat.append(cfproto_cfs)
            self.exectution_times['cfproto'] = time() - timer
            logging.debug(f'CFPROTO cfproto: {cfproto_cfs.shape[0]}')

        if 'carla' in self.list_of_explainers:
            # CARLA
            carla_cfs = self.__carla_generate_counterfactuals(query_instance=query_instance)
            if carla_cfs is None or len(carla_cfs) == 0:
                carla_cfs = counterfactuals.copy()
            to_concat.append(carla_cfs)
            logging.debug(f'CARLA generated: {carla_cfs.shape[0]}') 
        

        # Combine all generated counterfactuals
        counterfactuals = pd.concat([counterfactuals] + to_concat, ignore_index=True)

        logging.debug(counterfactuals['explainer'].tolist())

        counterfactuals[self.feature_name_to_predict] = self.__get_prediction_class_to_counterfactuals(counterfactuals)

        # Filter and save counterfactuals
        self.all_counterfactuals = counterfactuals.copy()
        self.counterfactuals_clipped = self.__clip_to_ranges(self.all_counterfactuals)
        self.valid_counterfactuals = self.__filter_only_valid(self.counterfactuals_clipped, query_instance)
        self.valid_actionable_counterfactuals = self.__filter_non_actionable(self.valid_counterfactuals, query_instance)

        return self.all_counterfactuals.reset_index(drop=True), self.valid_counterfactuals.reset_index(drop=True)

    # EXPLAINERS
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
        
        #logging.debug(train_df[self.feature_name_to_predict])

        self.dice_model = DiceModel(
            train_dataset=train_df,
            continuous_features=self.continuous_columns_names,
            categorical_features=self.categorical_columns_names,
            target=self.feature_name_to_predict,
            backend=backend,
            model=self.model_to_explain
        )
    
    def __dice_generate_counterfactuals(self, query_instance: pd.DataFrame, cfs_total: int = 20, desired_class: str | int = 'opposite') -> pd.DataFrame:
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
            cadex_max_epochs=50
        )

    def __fit_cfec_ece(self, fimap_load_models_date: str = '2023-03-11') -> None:
        self.cfec_model.fit(
            fimap_load_string=f'{self.dataset_shortname}_{self.model_backend_name}|{fimap_load_models_date}'
            )

    def __wachter_generate_counterfactuals(self, 
                                           query_instance: pd.DataFrame, 
                                           desired_class: int, 
                                           total_cfs: int = 10
                                           ) -> pd.DataFrame:
        '''Generate counterfactuals for query instance'''

        # Late import because of tensorflow version
        from explainers.alibi_wachter import AlibiWachter
        # load model specifically for alibi package methods because it does not support model loaded in eager mode ;(
        tf.compat.v1.disable_eager_execution()
        if isinstance(self.model_to_explain, tf.keras.Model):
            self.model_to_explain = tf.keras.models.load_model(self.model_path)

        query_shape = (1, self.train_dataset_ohe_normalized.shape[1])
        non_actionable_indices = ~np.array(self.actionable_mask_ohe, dtype='bool')
        
        # Freeze feature ranges on non-actionable features
        wachter_feature_ranges = (
            self.train_dataset_ohe_normalized.to_numpy().min(axis=0),
            self.train_dataset_ohe_normalized.to_numpy().max(axis=0),
        )
        
        query_instance_ohe_norm = self.transformer.transform_to_normalized_ohe(query_instance)
        wachter_feature_ranges[0][non_actionable_indices] = query_instance_ohe_norm.to_numpy()[0][non_actionable_indices]
        wachter_feature_ranges[1][non_actionable_indices] = query_instance_ohe_norm.to_numpy()[0][non_actionable_indices]
    

        self.wachter_model = AlibiWachter(self.model_to_explain, query_shape, self.transformer,
                                    target_proba=0.52, feature_ranges=wachter_feature_ranges,
                                    tolerance=0.01, target_class=desired_class
                                    )
        
        # Generate
        wachter_counterfactuals_df = self.wachter_model.generate_counterfactuals(query_instance_ohe_norm, total_cfs=total_cfs)

        return wachter_counterfactuals_df
    
    def __cem_generate_counterfactuals(self, query_instance: pd.DataFrame) -> None:
        from explainers.alibi_cem import AlibiCEM
        # load model specifically for alibi package methods because it does not support model loaded in eager mode ;(
        tf.compat.v1.disable_eager_execution()
        if isinstance(self.model_to_explain, tf.keras.Model):
            self.model_to_explain = tf.keras.models.load_model(self.model_path)
            
        cem_feature_ranges = (
            self.train_dataset_ohe_normalized.to_numpy().min(axis=0),
            self.train_dataset_ohe_normalized.to_numpy().max(axis=0),
        )
        non_actionable_indices = ~np.array(self.actionable_mask_ohe, dtype='bool')
        
        query_instance_ohe_norm = self.transformer.transform_to_normalized_ohe(query_instance)
        cem_feature_ranges[0][non_actionable_indices] = query_instance_ohe_norm.to_numpy()[0][non_actionable_indices]
        cem_feature_ranges[1][non_actionable_indices] = query_instance_ohe_norm.to_numpy()[0][non_actionable_indices]

        self.cem_model = AlibiCEM(model=self.model_to_explain, train_data_ohe_norm=self.train_dataset_ohe_normalized.to_numpy(), 
                        query_instance_shape=query_instance_ohe_norm.shape, feature_ranges=cem_feature_ranges, transformer=self.transformer
                        )

        cem_cfs_df = self.cem_model.generate_counterfactuals(query_instance, verbose=False)

        return cem_cfs_df

    def __cfproto_generate_counterfactuals(self, query_instance: pd.DataFrame, total_CFs: int = 10) -> None:
        from explainers.alibi_proto import AlibiProto
        # load model specifically for alibi package methods because it does not support model loaded in eager mode ;(
        tf.compat.v1.disable_eager_execution()
        if isinstance(self.model_to_explain, tf.keras.Model):
            self.model_to_explain = tf.keras.models.load_model(self.model_path)

        ranges = (
            np.zeros(query_instance.shape),
            np.ones(query_instance.shape)
        )

        logging.debug(f'SHAPE: {self.transformer.transform_to_normalized_ohe(query_instance).shape}')
        
        self.cfproto_model = AlibiProto(
                                model=self.model_to_explain, 
                                query_instance_shape=self.transformer.transform_to_normalized_ohe(query_instance).shape,
                                features_first_occurrence_indices=self.feature_first_occurrence_index_after_split, 
                                feature_value_counts=self.feature_value_counts, 
                                categorical_features_names=self.categorical_columns_names,
                                feature_ranges=ranges, transformer=self.transformer,
                            )

        self.cfproto_model.fit(self.train_dataset_ohe_normalized.to_numpy())

        cfproto_cfs_df = self.cfproto_model.generate_counterfactuals(query_instance, total_CFs=total_CFs)

        return cfproto_cfs_df

    def __carla_generate_counterfactuals(self, query_instance: pd.DataFrame) -> pd.DataFrame:
        
        from explainers.carla_models import CarlaModels

        # Reload model to avoid errors
        if isinstance(self.model_to_explain, tf.keras.Model):
                    self.model_to_explain = tf.keras.models.load_model(self.model_path)

        #query_instance_ohe_norm = self.transform_to_normalized_ohe(query_instance)

        self.carla_models = CarlaModels(
            train_dataset=self.train_dataset_pd,
            explained_model=self.model_to_explain,
            continous_columns=self.continuous_columns_names,
            categorical_columns=self.categorical_columns_names,
            nonactionable_columns=self.non_actionable_columns_names,
            target_feature_name=self.feature_name_to_predict,
            columns_order_ohe=self.features_order_after_split,
            transformer=self.transformer,
        )

        carla_cfs, execution_times_carla = self.carla_models.generate_counterfactuals(
            query_instance=query_instance,
        )

        for _explainer, _time in execution_times_carla.items():
            self.exectution_times[_explainer] = _time

        return carla_cfs

    def __clip_to_ranges(self, counterfactuals: pd.DataFrame) -> pd.DataFrame:
        '''Clip values outside of the defined ranges'''
        for feature, (lower, upper) in self.feature_ranges.items():
            #logging.debug(ranges)
           # lower, upper = ranges[0], ranges[1]
           counterfactuals.loc[counterfactuals[feature] < lower, feature] = lower
           counterfactuals.loc[counterfactuals[feature] > upper, feature] = upper
        return counterfactuals

    def __get_prediction_class_to_counterfactuals(self, counterfactuals: pd.DataFrame) -> pd.DataFrame:
        '''
        Assign prediction class to counterfactuals.
        '''
        cfs_norm_ohe = self.transformer.transform_to_normalized_ohe(counterfactuals).to_numpy().astype('float64')

        preds = np.argmax(self.model_to_explain.predict(cfs_norm_ohe), axis=1)

        return pd.DataFrame(preds, columns=[self.feature_name_to_predict])

    def __filter_only_valid(self, counterfactuals: pd.DataFrame, query_instance: pd.DataFrame) -> pd.DataFrame:
        '''
        Filter counterfactuals to retrieve only valid ones.
        '''
        assert self.feature_name_to_predict in counterfactuals.columns, 'Target feature should be in counterfactuals DataFrame'
        assert all(pd.notna(counterfactuals[self.feature_name_to_predict])), 'All counterfactuals should have their predicted class assigned'


        query_instance_ohe_norm = self.transformer.transform_to_normalized_ohe(query_instance).to_numpy().astype('float64')
        query_class = np.argmax(self.model_to_explain.predict(query_instance_ohe_norm), axis=1)

        mask = counterfactuals[self.feature_name_to_predict].astype('int') != int(query_class)

        return counterfactuals[mask].reset_index(drop=True)

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

        return counterfactuals[mask].reset_index(drop=True)

    # GETTERS
    def get_all_counterfactuals(self) -> pd.DataFrame:
        '''Get all counterfactuals that were found. No guarantees of validity and actionability.'''
        return self.all_counterfactuals

    def get_valid_counterfactuals(self) -> pd.DataFrame:
        '''Get all counterfactuals that were found and are guaranted to alter the prediction class after clipping to allowed ranges (if necessary).'''
        return self.valid_counterfactuals

    def get_valid_and_actionable_counterfactuals(self) -> pd.DataFrame:
        '''
        Get all counterfactuals that were found and are guaranted to alter 
        the prediction class and to be actionable according to constraints given.
        '''
        return self.valid_actionable_counterfactuals

    def get_quantitative_stats(self) -> Dict:
        '''
        Get quatitative stats for generated counterfactuals. 
        '''
        stats = dict()

        stats['dataset'] = self.dataset_shortname
        stats['all_cfs_count'] = int(self.all_counterfactuals.shape[0])
        stats['valid_cfs_count'] = int(self.valid_counterfactuals.shape[0])
        stats['valid_actionable_cfs_count'] = int(self.valid_actionable_counterfactuals.shape[0])
        stats['execution_times'] = self.exectution_times

        stats['explainers'] = dict()
        for explainer in np.unique(self.all_counterfactuals['explainer']):
            stats['explainers'][explainer] = dict()
            stats['explainers'][explainer]['all_cfs_count'] = int(
                self.all_counterfactuals[self.all_counterfactuals['explainer'] == explainer].shape[0]
                )
            stats['explainers'][explainer]['valid_cfs_count'] = int(
                self.valid_counterfactuals[self.valid_counterfactuals['explainer'] == explainer].shape[0]
                )
            stats['explainers'][explainer]['valid_actionable_cfs_count'] = int(
                self.valid_actionable_counterfactuals[self.valid_actionable_counterfactuals['explainer'] == explainer].shape[0]
                )

        return stats


if __name__ == '__main__':
    
    # Set logging
    logging.basicConfig(level=logging.DEBUG)
    
    train_dataset = pd.read_csv("data/adult.csv")
    dataset_name = 'adult'
    instance_to_explain_index = 5

    with open('data/adult_constraints.json', 'r') as f:
        constr = json.load(f)
    
    # TENSORFLOW
    mod_path = 'models/adult_NN/'
    explained_model = tf.keras.models.load_model(mod_path)

    train_dataset = train_dataset[constr['features_order_nonsplit']]
    query_instance = train_dataset.drop(columns="income")[instance_to_explain_index:instance_to_explain_index+1]

    _ensemble = Ensemble(
        train_dataset=train_dataset, constraints_config_dictionary=constr,
        model_to_explain=explained_model, model_path=mod_path,
        list_of_explainers=['cfproto'],
        )

    cfs = _ensemble.generate_counterfactuals(query_instance)

    logging.debug('----'*10)
    logging.debug('ALL', _ensemble.get_all_counterfactuals())
    logging.debug('----'*10)
    logging.debug('VALID', _ensemble.get_valid_counterfactuals())
    logging.debug('----'*10)
    logging.debug('VALID ACT', _ensemble.get_valid_and_actionable_counterfactuals())
    logging.debug('----'*10)


    