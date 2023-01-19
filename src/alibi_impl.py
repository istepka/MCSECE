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
    def __init__(self, model_path, model_type, query_instance_shape, target_proba=1.0, eps=None) -> None:
        '''
        model_path: full path
        model_type: sklearn or tensorflow
        query_instance_shape: shape of single instance e.g. (1, 85)
        eps: very important parameter - how each variable can be changed at one step. should be np.array of these values e.g. [0.1, 0.1, 1.0, 1.0]. 
        if this parameter is omitted then probably sklearn will work for very long time
        '''
        if eps is None:
            eps = 0.01

        if model_type == 'sklearn':
            with open(model_path ,'rb') as f:
                model = pickle.load(f)

            pred_fn = lambda x: np.array(model.predict_proba(x)[0])

            self.cf = Counterfactual(pred_fn, query_instance_shape, distance_fn='l1', target_proba=target_proba,
                                target_class='other', max_iter=40, early_stop=5, lam_init=0.0001,
                                max_lam_steps=4, tol=0.3, learning_rate_init=1.0,
                                feature_range=(0.0, 1.0), eps=eps, init='identity', 
                                decay=True, write_dir=None, debug=False)
        else: #TF
            model = tf.keras.models.load_model(model_path)

            self.cf = Counterfactual(model, query_instance_shape, distance_fn='l1', target_proba=target_proba,
                                target_class='other', max_iter=1000, early_stop=50, lam_init=0.1,
                                max_lam_steps=10, tol=0.01, learning_rate_init=0.1,
                                feature_range=(0.0, 1.0), eps=eps, init='identity',
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

   

if __name__ == '__main__':
    # SET THIS VARIABLE IF 
    explained_model_backend = 'sklearn' # 'sklearn' or 'tensorflow'

    # WARNING REMEMEBER TO CHANGE MANUALLY CFEC MODEL LOADING IF SOME CHANGES APPEAR 
    import numpy as np
    import pandas as pd
    import json
    from utils.transformations import min_max_normalization, inverse_min_max_normalization, transform_to_sparse, inverse_transform_to_sparse
    import warnings
    import tensorflow as tf
    import pickle

    warnings.filterwarnings('ignore', category=UserWarning) #Ignore sklearn "RF fitted with FeatureNames"

    train_dataset = pd.read_csv("data/adult.csv")
    dataset_name = 'adult'
    instance_to_explain_index = 9090

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
    train_dataset.columns

    query_instance = train_dataset.drop(columns="income")[instance_to_explain_index:instance_to_explain_index+1]

    # Transform dataset to sparse
    train_dataset_sparse = transform_to_sparse(
        _df=train_dataset.drop(columns="income"),
        original_df=train_dataset.drop(columns="income"),
        categorical_features=constr['categorical_features_nonsplit'],
        continuous_features=constr['continuous_features_nonsplit']
    )

    # Min-max normalization
    train_dataset_sparse_normalized = min_max_normalization(
        _df=train_dataset_sparse,
        original_df=train_dataset.drop(columns="income"),
        continuous_features=constr['continuous_features_nonsplit']
    )

    query_instance_sparse_normalized = train_dataset_sparse_normalized[instance_to_explain_index:instance_to_explain_index+1]
    
    continous = len(constr['continuous_features_nonsplit'])
    eps = np.array([[0.01] * continous + [1.0] * (len(train_dataset_sparse_normalized.columns) - continous)])
        

    if explained_model_backend == 'sklearn':
        wachter_model = AlibiWachter('models/adult_RF.pkl', 'sklearn', query_instance_sparse_normalized.shape, eps=eps)
    else:
        wachter_model = AlibiWachter('models/adult_NN/', 'tensorflow', query_instance_sparse_normalized.shape)
        
    explanation = wachter_model.generate_counterfactuals(query_instance_sparse_normalized)

    print(explanation)