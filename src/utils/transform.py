from typing import List, Tuple
import numpy as np
import numpy.typing as npt
import pandas as pd

from utils.transformations import *

class Transformer:
    '''
    Base Transformer - abstract class
    '''
    def __init__(self,
                 train_dataset: pd.DataFrame,
                 categorical_columns_names: List[str],
                 continuous_columns_names: List[str], 
                 feature_name_to_predict: str,
                 features_order_after_split: List[str]
                 ) -> None:
        self.train_dataset_pd = train_dataset
        self.categorical_columns_names = categorical_columns_names
        self.continuous_columns_names = continuous_columns_names
        self.feature_name_to_predict = feature_name_to_predict
        self.features_order_after_split = features_order_after_split

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