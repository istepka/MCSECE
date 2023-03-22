from typing import Dict, List
import numpy as np
import numpy.typing as npt
import pandas as pd
import logging
import tensorflow as tf

from alibi.explainers import CounterfactualProto
from sklearn.ensemble import RandomForestClassifier
from utils.transform import Transformer
from explainers.ExplainerBase import ExplainerBase

class AlibiProto(ExplainerBase):
    def __init__(self, model: tf.keras.Model | RandomForestClassifier, query_instance_shape: npt.NDArray,
        features_first_occurrence_indices: Dict[str, int], feature_value_counts: Dict[str, int], 
        categorical_features_names: List[str], feature_ranges: npt.NDArray, transformer: Transformer
    ) -> None:
        '''
        query_instance_shape: shape of single instance e.g. (1, 85) one-hot-encoded
        '''
        self.transformer = transformer
        
        # Create dictionary of {index: columns} for categorical ohe variables  
        self.cat_vars_ohe = {}
        for cat_feature in categorical_features_names:
            self.cat_vars_ohe[features_first_occurrence_indices[cat_feature]] = feature_value_counts[cat_feature]

        if isinstance(model, tf.keras.Model):
            self.cfproto = CounterfactualProto(model, query_instance_shape, 
                                                beta=.01, 
                                                cat_vars=self.cat_vars_ohe, 
                                                ohe=True, 
                                                max_iterations=500,
                                                feature_range=feature_ranges, 
                                                c_init=1.0, 
                                                c_steps=5,
                                            )
        else: #SKLEARN
            print('SKLEARN NOT SUPPORTED FOR ALIBIPROTO FOR NOW')

    def fit(self, training_data_ohe_norm: npt.NDArray, distance_metric_categorical: str = 'abdm', disc_perc=[25, 50, 75]) -> None:
        self.cfproto.fit(training_data_ohe_norm, d_type=distance_metric_categorical, disc_perc=disc_perc)

    def generate_counterfactuals(self, 
                                 query_instance: pd.DataFrame,
                                 total_CFs: int,
                                 ) -> pd.DataFrame:
        '''Generate counterfactuals for normalized ohe query instance'''
        
        query_instance_ohe_norm = self.transformer.transform_to_normalized_ohe(query_instance)
        
        explanation = self.cfproto.explain(query_instance_ohe_norm.to_numpy())
        
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

        cfproto_cfs_ohe_norm = pd.DataFrame(cfproto_counterfactuals, columns=self.transformer.features_order_after_split)

        cfproto_cfs_df = self.transformer.transform_from_norm_ohe(cfproto_cfs_ohe_norm)
        cfproto_cfs_df['explainer'] = 'cfproto'
        
        return cfproto_cfs_df
