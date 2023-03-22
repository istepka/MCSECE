import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf

from sklearn.ensemble import RandomForestClassifier
from alibi.explainers import CEM
from utils.transform import Transformer
from explainers.ExplainerBase import ExplainerBase

class AlibiCEM(ExplainerBase):
    def __init__(self, model: tf.keras.Model | RandomForestClassifier, train_data_ohe_norm: npt.NDArray,
    query_instance_shape: npt.NDArray, feature_ranges: npt.NDArray,  transformer: Transformer,
    eps: float = 0.05,
    ) -> None:

        self.transformer = transformer
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

    def generate_counterfactuals(self, 
                                 query_instance: pd.DataFrame,
                                 verbose: bool = False
                                 ) -> pd.DataFrame:
        '''Generate counterfactuals for normalized ohe query instance'''
        
        query_instance_ohe_norm = self.transformer.transform_to_normalized_ohe(query_instance)
        
        cem_explanation = self.cem.explain(query_instance_ohe_norm, verbose=verbose)
        
        if cem_explanation is None or cem_explanation.PN is None:
            return None
        cem_cfs_df_ohe_norm = pd.DataFrame(cem_explanation.PN, columns=self.transformer.features_order_after_split)

        cem_cfs_df = self.transformer.transform_from_norm_ohe(cem_cfs_df_ohe_norm)
        cem_cfs_df['explainer'] = 'cem'
        
        return cem_cfs_df