from typing import Dict, List
from modules.CFEC.cfec.explainers import Fimap, ECE, Cadex
from modules.CFEC.cfec.constraints import OneHot, ValueMonotonicity, ValueRange, Freeze
from utils.transformations import min_max_normalization, inverse_min_max_normalization, transform_to_sparse, inverse_transform_to_sparse
import pandas as pd
import tensorflow as tf
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Enable eager mode
tf.compat.v1.enable_eager_execution()

def train_supplementary_NN(train_data_normalized: np.ndarray, model_proba_predictions: np.ndarray) -> tf.keras.Model:
    print('Training supplementary model')
    input_shape = train_data_normalized.shape
    target_shape = model_proba_predictions.shape
    X_train, X_test, Y_train, Y_test = train_test_split(train_data_normalized, model_proba_predictions, test_size=0.3, random_state=2)
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input((input_shape[1],)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(2))
    model.add(tf.keras.layers.Softmax())

    model.compile(
        optimizer='rmsprop',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    model.fit(
        X_train, 
        Y_train,
        epochs=10,
        batch_size=256,
        validation_data=(X_test, Y_test),
        shuffle=True,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5)
        ],
        verbose=1
    )

    return model


class CfecEceModel:

    def __init__(self, train_data_normalized, constraints_dictionary, 
    model_path=None, model_backend=None, 
    fimap_load_s_g_full_id=None, fimap_save_s_q_prefix: str = None,
    columns_to_change: List[str] = None
    ) -> None:
        self.train_data = train_data_normalized
        self.constraints = None
        self.create_constraints(constraints_dictionary)

        if model_path and model_backend == 'tensorflow':
            model = tf.keras.models.load_model(model_path)
            model_predictions_proba = model.predict(self.train_data)
            model_predictions = np.argmax(model_predictions_proba, axis=1)
        else:
            with open(model_path, 'rb') as f:
                model = pickle.load(f) 
            model_predictions_proba = np.array(model.predict_proba(self.train_data)[0])
            model_predictions = np.argmax(model_predictions_proba, axis=1)

            # Train supplementary NN
            supp_model = train_supplementary_NN(self.train_data, model_predictions_proba)

        cadexes = []
        if model_path:      
            for n_changed in list(range(1, 15)):
                if model_backend == 'tensorflow':
                    cadex = Cadex(model, n_changed, constraints=self.constraints, optimizer=tf.keras.optimizers.legacy.Adam(0.05), max_epochs=20, columns_to_change=columns_to_change)
                    cadexes.append(cadex)
                if model_backend == 'sklearn':
                    cadex = Cadex(supp_model, n_changed, constraints=self.constraints, optimizer=tf.keras.optimizers.legacy.Adam(0.05), max_epochs=20, columns_to_change=columns_to_change)
                    cadexes.append(cadex)
        
        # Prepare fimap models
        fimaps = []
        fimap_hyperparameters = [
            (0.1, 0.001, 0.01),
            (0.1, 0.05, 0.5),
            (0.2, 0.01, 0.1),
            (0.2, 0.08, 0.8),
            (0.5, 0.001, 0.01)
        ]
        for i in range(len(fimap_hyperparameters)):
            tau, l1, l2 = fimap_hyperparameters[i]
            fimap = Fimap(tau, l1, l2, constraints=self.constraints, use_mapper=True, fimap_load_s_g_full_id=fimap_load_s_g_full_id, fimap_save_s_q_prefix=fimap_save_s_q_prefix)

            # Fit only if we do not want to load already trained models
            #if not fimap_load_s_g_full_id:
            fimap.fit(self.train_data, model_predictions)
            
            fimaps.append(fimap)

        self.ece = ECE(len(cadexes + fimaps), columns=list(self.train_data.columns), bces=fimaps + cadexes, dist=2, h=5, lambda_=0.001, n_jobs=1)

    def generate_counterfactuals(self, query_instance: pd.Series) -> pd.DataFrame:
        cfs = None
        list_cfs_explainers = None
        # try:
        cfs, list_cfs_explainers = self.ece.generate(query_instance)
        # except exception:
        #     print('No counterfactuals found (AttributeError). Return None') 
        return cfs, list_cfs_explainers

    
    
    def create_constraints(self, constraints_dictionary: Dict) -> List:
        
        constraints = []
        
        categorical = constraints_dictionary['categorical_features_nonsplit']
        continuous = constraints_dictionary['continuous_features_nonsplit']
        non_actionable = constraints_dictionary['non_actionable_features']
        feature_ranges = constraints_dictionary['feature_ranges']
        features_counts = constraints_dictionary['features_count_nonsplit']
        features_monotonicity = constraints_dictionary['features_monotonocity']
        feature_first_occurrence_after_split = constraints_dictionary['feature_first_occurrence_after_split']
        categorical_features_map_to_thier_splits = constraints_dictionary['categorical_features_map_to_thier_splits']

        self.nonsplit_columns_count = len(categorical) + len(continuous)

        # Index and length of each feature
        train_data_columns = categorical + continuous
        

        # # Freeze non_actionable features
        for feature in non_actionable:
            constraints.append(Freeze(categorical_features_map_to_thier_splits[feature])) # Append the split of the feature

        # OneHot constraints
        for feature in categorical:
            #if feature not in non_actionable: # CFEC does not allow 
            constraints.append(OneHot(
                    feature, 
                    feature_first_occurrence_after_split[feature], 
                    feature_first_occurrence_after_split[feature] + features_counts[feature] - 1
                    ))
        
        # ValueRange constraints
        for feature, (lower, upper) in feature_ranges.items():
            constraints.append(ValueRange([feature], lower, upper))

        # # ValueMonotonicity constraints
        # for feature, monotonicity in features_monotonicity.items():
        #     constraints.append(ValueMonotonicity([feature], monotonicity))

        print('Constraints: \n', constraints)

        self.constraints = constraints
        return constraints



if __name__ == '__main__':
   
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import json
    from utils.transformations import min_max_normalization, inverse_min_max_normalization, transform_to_sparse, inverse_transform_to_sparse
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning) #Ignore sklearn "RF fitted with FeatureNames"

    train_dataset = pd.read_csv("data/adult.csv")
    dataset_name = 'adult'
    instance_to_explain_index = 55

    with open('data/adult_constraints.json', 'r') as f:
        constr = json.load(f)

    train_dataset = train_dataset[constr['features_order_nonsplit']]
   
    query_instance = train_dataset.drop(columns="income")[instance_to_explain_index:instance_to_explain_index+1]

    all_counterfactuals = pd.DataFrame(columns=train_dataset.columns.tolist() + ['explainer'])
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

    train_dataset_sparse_normalized_subsample = train_dataset_sparse_normalized.sample(frac=1.0)


    cfec_model = CfecEceModel(
        train_data_normalized=train_dataset_sparse_normalized_subsample,
        constraints_dictionary=constr,
        model_path='models/adult_NN/',
        model_backend='tensorflow',
        # model_path='models/adult_RF.pkl',
        # model_backend='sklearn'
        fimap_load_s_g_full_id=f'adult_tensorflow|2023-01-17',
    )

    cfec_counterfactuals = cfec_model.generate_counterfactuals(query_instance=query_instance_sparse_normalized.iloc[0])
    # Inverse min-max normalization
    cfec_counterfactuals = inverse_min_max_normalization(
        _df=cfec_counterfactuals,
        original_df=train_dataset.drop(columns="income"),
        continuous_features=constr['continuous_features_nonsplit']
    )

    # Inverse transform to sparse
    cfec_counterfactuals = inverse_transform_to_sparse(
        sparse_df=cfec_counterfactuals,
        original_df=train_dataset.drop(columns="income"),
        categorical_features=constr['categorical_features_nonsplit'],
        continuous_features=constr['continuous_features_nonsplit']
    )
    
    print(cfec_counterfactuals)

