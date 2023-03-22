
from typing import List, Tuple
import pandas as pd
import tensorflow as tf
import json
from tqdm import tqdm
import time
import warnings

from modules.CARLA import carla
from carla.recourse_methods import GrowingSpheres, ActionableRecourse, Face
from carla.models.catalog import MLModelCatalog
from carla.data.catalog import CsvCatalog
from models.tfmodel import TFModelAdult
from utils.transform import Transformer

warnings.simplefilter(action='ignore', category=FutureWarning)

class CarlaModels:

    def __init__(self, train_dataset: pd.DataFrame, explained_model: tf.keras.Model,
        continous_columns: List[str], categorical_columns: List[str], 
        nonactionable_columns: List[str], target_feature_name: str, 
        columns_order_ohe: List[str], transformer: Transformer
    ) -> None:


        self.continous_columns = continous_columns
        self.categorical_columns = categorical_columns
        self.nonactionable_columns = nonactionable_columns
        self.target_feature_name = target_feature_name
        self.transformer = transformer
        self.data_catalog = CsvCatalog(dataset=train_dataset,
                            continuous=self.continous_columns,
                            categorical=self.categorical_columns,
                            immutables=self.nonactionable_columns,
                            target=self.target_feature_name)

        self.model = TFModelAdult(model=explained_model, data=self.data_catalog, columns_ohe_order=columns_order_ohe)


        # init explainers
        self.gs_explainer = GrowingSpheres(self.model)

        # AR only once because it gets the same cf
        ar_hyperparams = {
            "fs_size": 300, #default is 100
            "discretize": False,
            "sample": True,
        }
        self.ar_explainer = ActionableRecourse(self.model, ar_hyperparams)

    def generate_counterfactuals(self, query_instance: pd.DataFrame, 
        growing_spheres_restarts: int = 20, face_restarts: int = 10,
        ) -> Tuple[pd.DataFrame, dict]:
        '''
        Growing spheres is fast so can have more restarts than face
        
        returns: pd.DataFrame with counterfactuals, dict with execution times and explainers used
        '''
        execution_times = dict()
        to_concat = list()
        explainers_list = list()
        query_instance_ohe_norm = self.data_catalog.transform(query_instance)

        start = time.time()
        for _ in tqdm(range(growing_spheres_restarts), desc='Growing Spheres generating'):
            gs_cf = self.gs_explainer.get_counterfactuals(query_instance_ohe_norm)
            to_concat.append(gs_cf)
            explainers_list.append('growing-spheres')
        execution_times['growing-spheres'] = time.time() - start

        start = time.time()
        try: # Try catch because on rare ocasions something happens with coefficents inside recourse library (in test happened only 1 in 250 instances)
            ar_cf = self.ar_explainer.get_counterfactuals(query_instance_ohe_norm)
            to_concat.append(ar_cf)
            explainers_list.append('actionable-recourse')
        except AssertionError:
            print('Actionable Recourse threw an asserion error')
        execution_times['actionable-recourse'] = time.time() - start


        start = time.time()
        for i in tqdm(range(face_restarts), desc='FACE generating'):

            # Dynamically select fraction to prevent FACE from throwing errors because of the too small neighbourhood
            fraction = 0.05
            if self.data_catalog.df_train.shape[0] * 0.05 < 50:
                if self.data_catalog.df_train.shape[0] * 0.1 < 50:
                    fraction = 0.2
                else:
                    fraction = 0.1
            
            face_hyperparams = {
                'mode': 'knn',
                'fraction': fraction,
            }
            self.face_explainer = Face(self.model, face_hyperparams)
            face_cf = self.face_explainer.get_counterfactuals(query_instance_ohe_norm)
            to_concat.append(face_cf)
            explainers_list.append('face')
        execution_times['face'] = time.time() - start

        # Concatenate all counterfactuals and transform them back to original format
        carla_cfs_ohe_norm = pd.concat(to_concat, ignore_index=True).reset_index(drop=True)
        carla_cfs_ohe_norm = carla_cfs_ohe_norm.dropna()
        carla_cfs = self.transformer.transform_from_norm_ohe(carla_cfs_ohe_norm)
        carla_cfs['explainer'] = explainers_list
        
        return carla_cfs, execution_times

        

        