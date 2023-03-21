from typing import Dict, List
import numpy as np
import numpy.typing as npt
import pandas as pd
import pickle
from alibi.explainers import Counterfactual, CEM, CounterfactualProto
import tensorflow as tf
from utils.transformations import transform_to_sparse, inverse_min_max_normalization, min_max_normalization
from sklearn.ensemble import RandomForestClassifier


class AlibiCEM:
    def __init__(self, model: tf.keras.Model | RandomForestClassifier, train_data_ohe_norm: npt.NDArray,
    query_instance_shape: npt.NDArray, feature_ranges: npt.NDArray, 
    eps: float = 0.05,
    ) -> None:

        #tf.keras.backend.clear_session()
        shape = query_instance_shape  # instance shape
        clip = (-1000.,1000.) # gradient clip

        # Set perturbation size
        # _eps_tmp = np.ones(query_instance_shape[1]) # Categorical features can be perturbed only f
        # _eps_tmp[continous_features_indices] = eps #
        eps_cem = (eps, eps)
        

        if isinstance(model, RandomForestClassifier):
            mode = 'PN'
            update_num_grad = 2
            c_init = 15.  
            # Return probabilities for x
            cem_pred_fn = lambda x: np.array(model.predict_proba(x)[0])

            self.cem = CEM(cem_pred_fn, mode, shape, kappa=0.0, beta=0.1, feature_range=feature_ranges, 
                    update_num_grad=update_num_grad, clip=clip, no_info_val=-0.0, c_init=c_init,
                    c_steps=10, learning_rate_init=.1, max_iterations=10, eps=eps_cem
                    )
        else: # TF
            mode = 'PN'  
            kappa = .3 
            beta = .1 
            c_init = 10  
            c_steps = 10  
            max_iterations = 300
            lr_init = 1e-2  
            # initialize CEM explainer and explain instance
            self.cem = CEM(model, mode, shape, kappa=kappa, beta=beta, feature_range=feature_ranges,
                    max_iterations=max_iterations, c_init=c_init, c_steps=c_steps,
                    learning_rate_init=lr_init, clip=clip, no_info_val=0.0
                    )
        
        self.cem.fit(train_data_ohe_norm, no_info_type='median')

    def generate_counterfactuals(self, query_instance_ohe_norm: npt.NDArray, verbose: bool = False) -> npt.NDArray:
        '''Generate counterfactuals for normalized ohe query instance'''
        cem_explanation = self.cem.explain(query_instance_ohe_norm, verbose=verbose)
        return cem_explanation


class AlibiProto:
    def __init__(self, model: tf.keras.Model | RandomForestClassifier, query_instance_shape: npt.NDArray,
        features_first_occurrence_indices: Dict[str, int], feature_value_counts: Dict[str, int], 
        categorical_features_names: List[str], feature_ranges: npt.NDArray
    ) -> None:
        '''
        query_instance_shape: shape of single instance e.g. (1, 85)
        '''

        # Create dictionary of {index: columns} for categorical ohe variables  
        cat_vars_ohe = {}
        for cat_feature in categorical_features_names:
            cat_vars_ohe[features_first_occurrence_indices[cat_feature]] = feature_value_counts[cat_feature]


        if isinstance(model, tf.keras.Model):
            self.cfproto = CounterfactualProto(model, query_instance_shape, 
                beta=.01, 
                cat_vars=cat_vars_ohe, 
                ohe=True, 
                max_iterations=500,
                feature_range=feature_ranges, 
                c_init=1.0, 
                c_steps=5,
            )
        else: #SKLEARN
            # pred_fn = lambda x: np.array(model.predict_proba(x)[0])
            # self.cf = CounterFactualProto(pred_fn)
            print('SKLEARN NOT SUPPORTED FOR ALIBIPROTO FOR NOW')

    def fit(self, training_data_ohe_norm: npt.NDArray, distance_metric_categorical: str = 'abdm', disc_perc=[25, 50, 75]) -> None:
        self.cfproto.fit(training_data_ohe_norm, d_type=distance_metric_categorical, disc_perc=disc_perc)

    def generate_counterfactuals(self, query_instance_ohe_norm: npt.NDArray) -> npt.NDArray:
        '''Generate counterfactuals for normalized ohe query instance'''
        explanation = self.cfproto.explain(query_instance_ohe_norm)
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