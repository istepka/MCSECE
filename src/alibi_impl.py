import numpy as np
import pandas as pd
import pickle
from alibi.explainers import Counterfactual 
import tensorflow as tf
from utils.transformations import transform_to_sparse, inverse_min_max_normalization, min_max_normalization

tf.compat.v1.disable_eager_execution()

# # Define the model
# model = tf.keras.models.load_model('../models/adult_NN/')

class AlibiWachter:
    def __init__(self, model_path, model_type, query_instance_shape) -> None:
        '''
        model_path: full path
        model_type: sklearn or tensorflow
        query_instance_shape: shape of single instance e.g. (1, 85)
        '''

        if model_type == 'sklearn':
            with open(model_path ,'rb') as f:
                model = pickle.load(f)

            pred_fn = lambda x: np.array(model.predict_proba(x)[0])

            self.cf = Counterfactual(pred_fn, query_instance_shape, distance_fn='l1', target_proba=1.0,
                                target_class='other', max_iter=100, early_stop=50, lam_init=1e-3,
                                max_lam_steps=15, tol=0.4, learning_rate_init=0.1,
                                feature_range=(0, 1), eps=0.1, init='identity',
                                decay=True, write_dir=None, debug=False)
        else: #TF
            model = tf.keras.models.load_model(model_path)

            self.cf = Counterfactual(model, query_instance_shape, distance_fn='l1', target_proba=1.0,
                                target_class='other', max_iter=1000, early_stop=100, lam_init=1e-1,
                                max_lam_steps=15, tol=0.4, learning_rate_init=0.1,
                                feature_range=(0, 1), eps=0.1, init='identity',
                                decay=True, write_dir=None, debug=False)

    def generate_counterfactuals(self, query_instance_norm: pd.DataFrame | np.ndarray):
        '''Generate counterfactuals for normalized query instance'''
        if type(query_instance_norm) == pd.DataFrame:
            explanation = self.cf.explain(query_instance_norm.to_numpy())
        else:
            explanation = self.cf.explain(query_instance_norm)
        return explanation

class AlibiProto:
    def __init__(self, model_path, model_type, query_instance_shape) -> None:
        '''
        model_path: full path
        model_type: sklearn or tensorflow
        query_instance_shape: shape of single instance e.g. (1, 85)
        '''

        if model_type == 'sklearn':
            with open(model_path ,'rb') as f:
                model = pickle.load(f)

            pred_fn = lambda x: np.array(model.predict_proba(x)[0])

            self.cf = Counterfactual(pred_fn, query_instance_shape, distance_fn='l1', target_proba=1.0,
                                target_class='other', max_iter=100, early_stop=50, lam_init=1e-3,
                                max_lam_steps=15, tol=0.4, learning_rate_init=0.1,
                                feature_range=(0, 1), eps=0.3, init='identity',
                                decay=True, write_dir=None, debug=False)
        else: #TF
            model = tf.keras.models.load_model(model_path)

            self.cf = Counterfactual(model, query_instance_shape, distance_fn='l1', target_proba=1.0,
                                target_class='other', max_iter=1000, early_stop=50, lam_init=1e-1,
                                max_lam_steps=15, tol=0.4, learning_rate_init=0.1,
                                feature_range=(0, 1), eps=0.1, init='identity',
                                decay=True, write_dir=None, debug=False)

    def generate_counterfactuals(self, query_instance_norm: pd.DataFrame | np.ndarray):
        '''Generate counterfactuals for normalized query instance'''
        if type(query_instance_norm) == pd.DataFrame:
            explanation = self.cf.explain(query_instance_norm.to_numpy())
        else:
            explanation = self.cf.explain(query_instance_norm)
        return explanation

   