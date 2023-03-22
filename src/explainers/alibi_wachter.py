from typing import Dict, List
import numpy as np
import numpy.typing as npt
import pandas as pd
from alibi.explainers import Counterfactual
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier

from explainers.ExplainerBase import ExplainerBase
from utils.transform import Transformer

class AlibiWachter(ExplainerBase):
    def __init__(self, model: tf.keras.Model | RandomForestClassifier, query_instance_shape: npt.NDArray, 
                transformer: Transformer, target_proba=0.5, eps=None, feature_ranges=None,
                max_iter=100, max_lam_steps=10, lam_init=0.0001, learning_rate_init=1.0,
                early_stop = 5, tolerance=0.01, target_class: int | str = 'other'
        ) -> None:
        '''
        model: blackbox model - currently supported tensorflow and RandomForest from sklearn
        query_instance_shape: shape of single instance e.g. (1, 85)
        eps: very important parameter - how each variable can be changed at one step. should be np.array of these values e.g. [0.1, 0.1, 1.0, 1.0]. 
        if this parameter is omitted then probably sklearn will work for very long time
        '''
        self.transformer = transformer
        self.model = model
        
        if feature_ranges is None:
            feature_ranges = (0.0, 1.0)

        if isinstance(model, RandomForestClassifier):
            pred_fn = lambda x: np.array(model.predict_proba(x)[0])

            self.cf = Counterfactual(pred_fn, query_instance_shape, distance_fn='l1', target_proba=target_proba,
                                target_class=target_class, max_iter=max_iter, early_stop=early_stop, lam_init=lam_init,
                                max_lam_steps=max_lam_steps, tol=tolerance, learning_rate_init=learning_rate_init,
                                feature_range=feature_ranges, init='identity', 
                                decay=True, write_dir=None, debug=False)
        else: #TF
            self.cf = Counterfactual(model, query_instance_shape, distance_fn='l1', target_proba=target_proba,
                                target_class=target_class, max_iter=1000, early_stop=200, lam_init=0.1,
                                max_lam_steps=10, tol=tolerance, learning_rate_init=0.1,
                                feature_range=feature_ranges, init='identity',
                                decay=True, write_dir=None, debug=False)

    def generate_counterfactuals(self, 
                                 query_instance: pd.DataFrame,
                                 total_cfs: int = 10,
                                 ) -> pd.DataFrame:
        '''
        Generate counterfactuals for query instance
        
        `total_cfs` - number of counterfactuals to generate. 
        If more than 1 then random sample is taken from all found 
        counterfactuals in the optimization process
        '''
        query_instance_ohe_norm = self.transformer.transform_to_normalized_ohe(query_instance)
        
        if type(query_instance_ohe_norm) == pd.DataFrame:
            explanation = self.cf.explain(query_instance_ohe_norm.to_numpy())
        else:
            explanation = self.cf.explain(query_instance_ohe_norm)
            
            
        wachter_counterfactuals = []
        
        if total_cfs > 1:
            # Get counterfactuals from the optimization process
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
        else:
            wachter_counterfactuals = explanation.cf['X']

        # Transform to original dataframe format
        wachter_counterfactuals_df_ohe_norm = pd.DataFrame(wachter_counterfactuals, columns=self.transformer.features_order_after_split)
        wachter_counterfactuals_df = self.transformer.transform_from_norm_ohe(wachter_counterfactuals_df_ohe_norm)
        wachter_counterfactuals_df['explainer'] = 'wachter'
            
        return wachter_counterfactuals_df
    
