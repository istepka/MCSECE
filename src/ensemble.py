# Libs
from typing import Dict, List
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from time import time

# Own
from dice import DiceModel 
from cfec_ece import CfecEceModel
from utils.transformations import min_max_normalization, inverse_min_max_normalization, transform_to_sparse, inverse_transform_to_sparse

class Ensemble:

    def __init__(self, train_dataset: pd.DataFrame, constraints_config_dictionary: Dict,
        model_to_explain: tf.keras.Model | RandomForestClassifier, model_path: str,
        list_of_explainers: List[str] = ['dice', 'cfec', 'alibi', 'carla'],
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
        self.exectution_times: Dict[str, int] = dict() 

    def generate_counterfactuals(self, query_instance: pd.DataFrame) -> pd.DataFrame:
        '''
        Can ask only for one query instance at a time.
        Has to be in pd.DataFrame format.

        Return: valid and actionable countefactuals as data frame
        '''
        assert query_instance.shape[0] == 1, 'Only one query instance at a time supported!'

        counterfactuals = pd.DataFrame(columns=self.all_features_without_target + ['explainer'])

        query_instance = query_instance[self.features_order_before_split_without_target]

        query_instance_norm_ohe = self.transform_to_normalized_ohe(query_instance)

        query_instance_class_int = int(np.argmax(self.model_to_explain.predict(query_instance_norm_ohe.to_numpy().astype("float64"))))
        desired_class = 1 - query_instance_class_int
        print(f'X class: {query_instance_class_int}')
        print(f'Desired cf class: {desired_class}')

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
            print(f'Dice generated: {dice_cfs.shape[0]}')


        if 'cfec' in self.list_of_explainers:
            # generate CFEC
            timer = time()
            cfec_cfs_norm_ohe, cfec_explainers = self.cfec_model.generate_counterfactuals(
                query_instance=query_instance_norm_ohe.iloc[0]
            )
            if cfec_cfs_norm_ohe is not None and len(cfec_cfs_norm_ohe) > 0:
                cfec_cfs = self.transform_from_norm_ohe(cfec_cfs_norm_ohe)
                cfec_cfs['explainer'] = cfec_explainers
                to_concat.append(cfec_cfs)
            else:
                cfec_cfs = counterfactuals.copy()
            self.exectution_times['cfec'] = time() - timer
            print(f'CFEC generated: {cfec_cfs.shape[0]}')


        if 'alibi' in self.list_of_explainers:
            # generate Wachter
            timer = time()
            wachter_cfs = self.__wachter_generate_counterfactuals(query_instace=query_instance, desired_class=desired_class)
            if wachter_cfs is None or len(wachter_cfs) == 0:
                wachter_cfs = counterfactuals.copy()
            wachter_cfs['explainer'] = 'wachter'
            to_concat.append(wachter_cfs)
            self.exectution_times['wachter'] = time() - timer
            print(f'Wachter generated: {wachter_cfs.shape[0]}')

            # generate CEM
            timer = time()
            cem_cfs = self.__cem_generate_counterfactuals(query_instance)
            if cem_cfs is None or len(cem_cfs) == 0:
                cem_cfs = counterfactuals.copy()
            cem_cfs['explainer'] = 'cem'
            to_concat.append(cem_cfs)
            self.exectution_times['cem'] = time() - timer
            print(f'CEM generated: {cem_cfs.shape[0]}')

            # generate CfProto
            timer = time()
            cfproto_cfs = self.__cfproto_generate_counterfactuals(query_instance)
            if cfproto_cfs is None or len(cfproto_cfs) == 0:
                cfproto_cfs = counterfactuals.copy()
            cfproto_cfs['explainer'] = 'cfproto'
            to_concat.append(cfproto_cfs)
            self.exectution_times['cfproto'] = time() - timer
            print(f'CFPROTO cfproto: {cfproto_cfs.shape[0]}')

        if 'carla' in self.list_of_explainers:
            # CARLA
            carla_cfs = self.__carla_generate_counterfactuals(query_instance=query_instance)
            if carla_cfs is None or len(carla_cfs) == 0:
                carla_cfs = counterfactuals.copy()
            to_concat.append(carla_cfs)
            print(f'CARLA generated: {carla_cfs.shape[0]}') 
        

        # Combine all generated counterfactuals
        counterfactuals = pd.concat([counterfactuals] + to_concat, ignore_index=True)

        print(counterfactuals['explainer'].tolist())

        counterfactuals[self.feature_name_to_predict] = self.__get_prediction_class_to_counterfactuals(counterfactuals)

        # Filter and save counterfactuals
        self.all_counterfactuals = counterfactuals.copy()
        self.counterfactuals_clipped = self.__clip_to_ranges(self.all_counterfactuals)
        self.valid_counterfactuals = self.__filter_only_valid(self.counterfactuals_clipped, query_instance)
        self.valid_actionable_counterfactuals = self.__filter_non_actionable(self.valid_counterfactuals, query_instance)

        return self.valid_actionable_counterfactuals.reset_index(drop=True)

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
        
        #print(train_df[self.feature_name_to_predict])

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

    def __fit_cfec_ece(self, fimap_load_models_date: str = '2023-01-26') -> None:
        self.cfec_model.fit(
            fimap_load_string=f'{self.dataset_shortname}_{self.model_backend_name}|{fimap_load_models_date}'
            )

    def __wachter_generate_counterfactuals(self, query_instace: pd.DataFrame, desired_class: int, total_cfs: int = 10) -> pd.DataFrame:
        
        from alibi_impl import AlibiWachter

        query_instance_ohe_norm = self.transform_to_normalized_ohe(query_instace)

        # Freeze feature ranges on non-actionable features
        wachter_feature_ranges = (
            self.train_dataset_ohe_normalized.to_numpy().min(axis=0),
            self.train_dataset_ohe_normalized.to_numpy().max(axis=0),
        )
        non_actionable_indices = ~np.array(self.actionable_mask_ohe, dtype='bool')
        wachter_feature_ranges[0][non_actionable_indices] = query_instance_ohe_norm.to_numpy()[0][non_actionable_indices]
        wachter_feature_ranges[1][non_actionable_indices] = query_instance_ohe_norm.to_numpy()[0][non_actionable_indices]

        query_shape = (1, self.train_dataset_ohe_normalized.shape[1])

        # Initialize explainer
        if self.model_backend_name == 'sklearn':
            self.wachter_model = AlibiWachter(self.model_to_explain, query_shape, 
                                        feature_ranges=wachter_feature_ranges, max_iter=100, max_lam_steps=10, 
                                        lam_init=0.001, learning_rate_init=0.1, early_stop=50, tolerance=0.01, 
                                        target_proba=0.6, target_class=desired_class
                                        )
        else:
            tf.compat.v1.disable_eager_execution()
            # load model specifically for Wachter because it does not support model loaded in eager mode ;(
            if isinstance(self.model_to_explain, tf.keras.Model):
                self.model_to_explain = tf.keras.models.load_model(self.model_path)

            self.wachter_model = AlibiWachter(self.model_to_explain, query_shape, 
                                        target_proba=0.52, feature_ranges=wachter_feature_ranges,
                                        tolerance=0.01, target_class=desired_class
                                        )

        # Generate
        explanation = self.wachter_model.generate_counterfactuals(query_instance_ohe_norm)

        # Get counterfactuals from the optimization process
        wachter_counterfactuals = []
        for _, lst in explanation['data']['all'].items():
            if lst:
                for cf in lst:
                    wachter_counterfactuals.append(cf['X'])
        
        # If no counterfactuals found return none
        if len(wachter_counterfactuals) == 0:
            return None

        # Reshape to (n, features)
        wachter_counterfactuals = np.array(wachter_counterfactuals).reshape(-1, query_instance_ohe_norm.shape[1])
        
        # Get random sample from all cfs to get desired number 
        _indices_to_take = np.random.permutation(wachter_counterfactuals.shape[0])[0:total_cfs-1]
        wachter_counterfactuals = wachter_counterfactuals[_indices_to_take, :]

        # Concat sample with the one counterfactual that wachter chose as best found
        wachter_counterfactuals = np.concatenate([wachter_counterfactuals, explanation.cf['X']], axis=0)

        # Transform to original dataframe format
        wachter_counterfactuals_df_ohe_norm = pd.DataFrame(wachter_counterfactuals, columns=self.features_order_after_split)
        wachter_counterfactuals_df = self.transform_from_norm_ohe(wachter_counterfactuals_df_ohe_norm)
        wachter_counterfactuals_df['explainer'] = 'wachter'

        return wachter_counterfactuals_df
    
    def __cem_generate_counterfactuals(self, query_instace: pd.DataFrame) -> None:
        from alibi_impl import AlibiCEM

        query_instance_ohe_norm = self.transform_to_normalized_ohe(query_instace)

        cem_feature_ranges = (
            self.train_dataset_ohe_normalized.to_numpy().min(axis=0),
            self.train_dataset_ohe_normalized.to_numpy().max(axis=0),
        )
        non_actionable_indices = ~np.array(self.actionable_mask_ohe, dtype='bool')
        cem_feature_ranges[0][non_actionable_indices] = query_instance_ohe_norm.to_numpy()[0][non_actionable_indices]
        cem_feature_ranges[1][non_actionable_indices] = query_instance_ohe_norm.to_numpy()[0][non_actionable_indices]

        self.cem_model = AlibiCEM(model=self.model_to_explain, train_data_ohe_norm=self.train_dataset_ohe_normalized.to_numpy(), 
                        query_instance_shape=query_instance_ohe_norm.shape, feature_ranges=cem_feature_ranges,
                        )

        cem_cfs = self.cem_model.generate_counterfactuals(query_instance_ohe_norm.to_numpy(), verbose=False)
        cem_cfs_df_ohe_norm = pd.DataFrame(cem_cfs.PN, columns=self.features_order_after_split)

        cem_cfs_df = self.transform_from_norm_ohe(cem_cfs_df_ohe_norm)
        cem_cfs_df['explainer'] = 'cem'

        return cem_cfs_df

    def __cfproto_generate_counterfactuals(self, query_instance: pd.DataFrame, total_CFs: int = 10) -> None:
        from alibi_impl import AlibiProto
        query_instance_ohe_norm = self.transform_to_normalized_ohe(query_instance)

        # cfprot_feature_ranges = (
        #     self.train_dataset_ohe_normalized.to_numpy().min(axis=0),
        #     self.train_dataset_ohe_normalized.to_numpy().max(axis=0),
        # )
        # non_actionable_indices = ~np.array(self.actionable_mask_ohe, dtype='bool')
        # cfprot_feature_ranges[0][non_actionable_indices] = query_instance_ohe_norm.to_numpy()[0][non_actionable_indices]
        # cfprot_feature_ranges[1][non_actionable_indices] = query_instance_ohe_norm.to_numpy()[0][non_actionable_indices]
        # cfprot_feature_ranges = (cfprot_feature_ranges[0].reshape(1,-1), cfprot_feature_ranges[1].reshape(1,-1))

        frozen_indices = [query_instance.columns.tolist().index(feat) for feat in self.non_actionable_columns_names]

        ranges = (
            np.zeros(query_instance.shape),
            np.ones(query_instance.shape)
        )
        # Freeze possible feature changes on frozen features
        #ranges[1][:, frozen_indices] = 0

        self.cfproto_model = AlibiProto(
                                model=self.model_to_explain, 
                                query_instance_shape=query_instance_ohe_norm.shape,
                                features_first_occurrence_indices=self.feature_first_occurrence_index_after_split, 
                                feature_value_counts=self.feature_value_counts, 
                                categorical_features_names=self.categorical_columns_names,
                                feature_ranges=ranges
                            )

        self.cfproto_model.fit(self.train_dataset_ohe_normalized.to_numpy())

        explanation = self.cfproto_model.generate_counterfactuals(query_instance_ohe_norm.to_numpy())

        # If no coutnterfactuals found
        if explanation is None or len(explanation['data']['all']) == 0:
            return None

         # Get counterfactuals from the optimization process
        cfproto_counterfactuals = []
        for _, lst in explanation['data']['all'].items():
            if lst:
                for cf in lst:
                    cfproto_counterfactuals.append(cf)

        # Reshape to (n, features)
        cfproto_counterfactuals = np.array(cfproto_counterfactuals).reshape(-1, query_instance_ohe_norm.shape[1])
        
        # Get random sample from all cfs to get desired number 
        _indices_to_take = np.random.permutation(cfproto_counterfactuals.shape[0])[0:total_CFs-1]
        cfproto_counterfactuals = cfproto_counterfactuals[_indices_to_take, :]

        # Concat sample with the one counterfactual that wachter chose as best found
        cfproto_counterfactuals = np.concatenate([cfproto_counterfactuals, explanation.cf['X']], axis=0)

        cfproto_cfs_ohe_norm = pd.DataFrame(cfproto_counterfactuals, columns=self.features_order_after_split)

        cfproto_cfs_df = self.transform_from_norm_ohe(cfproto_cfs_ohe_norm)
        cfproto_cfs_df['explainer'] = 'cfproto'

        return cfproto_cfs_df

    def __carla_generate_counterfactuals(self, query_instance: pd.DataFrame) -> pd.DataFrame:
        
        from carla_models import CarlaModels

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
        )

        carla_cfs_ohe_norm, explainers, execution_times_carla = self.carla_models.generate_counterfactuals(
            query_instance=query_instance,
        )

        for _explainer, _time in execution_times_carla.items():
            self.exectution_times[_explainer] = _time

        carla_cfs_ohe_norm['explainer'] = explainers
        carla_cfs_ohe_norm = carla_cfs_ohe_norm.dropna()
        explainers = carla_cfs_ohe_norm['explainer']

        carla_cfs = self.transform_from_norm_ohe(carla_cfs_ohe_norm.drop(columns='explainer'))
        carla_cfs['explainer'] = explainers
        return carla_cfs

    # UTILITY FUNCTIONS 
    def transform_to_normalized_ohe(self, query_instance: pd.DataFrame) -> pd.DataFrame:
        '''Transform original dataframe into one-hot-encoded and normalized form.'''
        query_instance_ohe = transform_to_sparse(
            _df = query_instance,
            original_df=self.train_dataset_pd.drop(columns=self.feature_name_to_predict),
            categorical_features=self.categorical_columns_names,
            continuous_features=self.continuous_columns_names
        )

        query_instance_ohe_norm = min_max_normalization(
            _df=query_instance_ohe,
            original_df=self.train_dataset_pd.drop(columns=self.feature_name_to_predict),
            continuous_features=self.continuous_columns_names
        )

        return query_instance_ohe_norm[self.features_order_after_split] # Make sure that correct order is mantaineed

    def transform_from_norm_ohe(self, query_instance_norm_ohe: pd.DataFrame) -> pd.DataFrame:
        '''Transform from one-hot-encoded normalized form into original dataframe.'''
        query_instance_ohe = inverse_min_max_normalization(
            _df=query_instance_norm_ohe,
            original_df=self.train_dataset_pd.drop(columns=self.feature_name_to_predict),
            continuous_features=self.continuous_columns_names
        )

        query_instance = inverse_transform_to_sparse(
            sparse_df=query_instance_ohe,
            original_df=self.train_dataset_pd.drop(columns=self.feature_name_to_predict),
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
    explained_model_backend = 'tensorflow' # 'sklearn' or 'tensorflow'


    from sklearn.model_selection import train_test_split
    import json
    import warnings
    import pickle

    warnings.filterwarnings('ignore', category=UserWarning) #Ignore sklearn "RF fitted with FeatureNames"

    train_dataset = pd.read_csv("data/adult.csv")
    dataset_name = 'adult'
    instance_to_explain_index = 5

    with open('data/adult_constraints.json', 'r') as f:
        constr = json.load(f)
    
    if explained_model_backend == 'sklearn':
        # SKLEARN
        mod_path = 'models/adult_RF.pkl'
        with open(mod_path, 'rb') as f:
            explained_model = pickle.load(f)
    else: 
        # TENSORFLOW
        mod_path = 'models/adult_NN/'
        explained_model = tf.keras.models.load_model(mod_path)


    train_dataset = train_dataset[constr['features_order_nonsplit']]
    query_instance = train_dataset.drop(columns="income")[instance_to_explain_index:instance_to_explain_index+1]


    enseble = Ensemble(
        train_dataset=train_dataset, constraints_config_dictionary=constr,
        model_to_explain=explained_model, model_path=mod_path,
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

    