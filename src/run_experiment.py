import json
import warnings
import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime

from ensemble import Ensemble
from utils.scores import get_scores
from utils.transformations import min_max_normalization, transform_to_sparse

date = datetime.now().strftime(r"%Y-%m-%d")  

TRAIN_DATASET_PATH = 'data/adult_train.csv'
TEST_DATASET_PATH = 'data/adult_test.csv'

DATASET_NAME = 'adult'
CONSTRAINTS_PATH = 'data/adult_constraints.json'

MODEL_PATH = 'models/adult_NN/'
EXPLAINED_MODEL_BACKEND = 'tensorflow'

INDEX_TO_EXPLAIN = 12

SAVE_PATH = f'experiments/scores/{DATASET_NAME}_{EXPLAINED_MODEL_BACKEND}_i{INDEX_TO_EXPLAIN}_{date}.csv'


#----------
PREFERENCES_RANKING = [0, 4, 2, 3, 5, 1]
K_NEIGHBORS_FEASIB = 3
K_NEIGHBORS_DISCRIMINATIVE = 9


if __name__ == '__main__':
    explained_model_backend = 'tensorflow' # 'sklearn' or 'tensorflow'


    warnings.filterwarnings('ignore', category=UserWarning) #Ignore sklearn "RF fitted with FeatureNames"
    warnings.filterwarnings('ignore', category=FutureWarning) #Ignore sklearn "RF fitted with FeatureNames"

    train_dataset = pd.read_csv(TRAIN_DATASET_PATH)
    test_dataset = pd.read_csv(TEST_DATASET_PATH)

    

    with open(CONSTRAINTS_PATH, 'r') as f:
        constr = json.load(f)
    
    if EXPLAINED_MODEL_BACKEND == 'sklearn':
        # SKLEARN
        with open(MODEL_PATH, 'rb') as f:
            explained_model = pickle.load(f)
    else: 
        # TENSORFLOW
        explained_model = tf.keras.models.load_model(MODEL_PATH)

    # Make sure that order is correct
    train_dataset = train_dataset[constr['features_order_nonsplit']]


    test_dataset_no_target = test_dataset.drop(columns='income', inplace=False)

    query_instance = test_dataset_no_target[INDEX_TO_EXPLAIN:INDEX_TO_EXPLAIN+1]

    print(query_instance)

    enseble = Ensemble(
        train_dataset=train_dataset, constraints_config_dictionary=constr,
        model_to_explain=explained_model, model_path=MODEL_PATH,
        )

    cfs = enseble.generate_counterfactuals(query_instance)

    print('----'*10)
    print('ALL', enseble.get_all_counterfactuals())
    print('----'*10)
    print('VALID', enseble.get_valid_counterfactuals())
    print('----'*10)
    print('VALID ACT', enseble.get_valid_and_actionable_counterfactuals())
    print('----'*10)


    target_feature_name = constr['target_feature']

    continous_indices = list()
    categorical_indices = list()
    cols = test_dataset_no_target.columns.tolist()

    for col in constr['continuous_features_nonsplit']:
        continous_indices += [cols.index(col)]

    for col in constr['categorical_features_nonsplit']:
        categorical_indices += [cols.index(col)]


    train_dataset_no_target = train_dataset.drop(columns=target_feature_name)
    train_dataset_no_target_ohe = transform_to_sparse(
        _df = train_dataset_no_target,
        original_df=train_dataset_no_target,
        categorical_features=constr['categorical_features_nonsplit'],
        continuous_features=constr['continuous_features_nonsplit']
    )

    train_dataset_no_target_ohe_norm = min_max_normalization(
        _df=train_dataset_no_target_ohe,
        original_df=train_dataset_no_target,
        continuous_features=constr['continuous_features_nonsplit']
    )

    train_dataset_no_target_ohe_norm = train_dataset_no_target_ohe_norm[constr['features_order_after_split']]
    
    if EXPLAINED_MODEL_BACKEND == 'sklearn':
        # SKLEARN
        with open(MODEL_PATH, 'rb') as f:
            explained_model = pickle.load(f)
    else: 
        # TENSORFLOW
        explained_model = tf.keras.models.load_model(MODEL_PATH)

    train_preds = np.argmax(explained_model.predict(train_dataset_no_target_ohe_norm.to_numpy()), axis=1)

    scores = get_scores(
        cfs=cfs.drop(columns=[target_feature_name, 'explainer']).to_numpy().astype('<U11'), 
        cf_predicted_classes=cfs[target_feature_name].to_numpy(),
        training_data=train_dataset.drop(columns=target_feature_name).to_numpy().astype('<U11'),
        training_data_predicted_classes=train_preds,
        x = query_instance.to_numpy()[0].astype('<U11'),
        x_predicted_class= test_dataset[INDEX_TO_EXPLAIN:INDEX_TO_EXPLAIN+1][target_feature_name],
        continous_indices=continous_indices,
        categorical_indices=categorical_indices,
        preferences_ranking=PREFERENCES_RANKING,
        k_neighbors_feasib=K_NEIGHBORS_FEASIB,
        k_neighbors_discriminative=K_NEIGHBORS_DISCRIMINATIVE
        )

    scores['explainer'] = cfs['explainer']


    scores.to_csv(SAVE_PATH, index=False)

    print(scores)
