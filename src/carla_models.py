
from typing import List
import pandas as pd
import tensorflow as tf
from modules.CARLA import carla
from carla.recourse_methods import GrowingSpheres, ActionableRecourse, Face
from carla.models.catalog import MLModelCatalog
import warnings
from carla.data.catalog import CsvCatalog
import json
from tqdm import tqdm

from models.tfmodel import TFModelAdult

warnings.simplefilter(action='ignore', category=FutureWarning)


class CarlaModels:

    def __init__(self, train_dataset: pd.DataFrame, explained_model: tf.keras.Model,
        continous_columns: List[str], categorical_columns: List[str], 
        nonactionable_columns: List[str], target_feature_name: str, 
        columns_order_ohe: List[str],
    ) -> None:


        self.continous_columns = continous_columns
        self.categorical_columns = categorical_columns
        self.nonactionable_columns = nonactionable_columns
        self.target_feature_name = target_feature_name

        self.data_catalog = CsvCatalog(dataset=train_dataset,
                            continuous=self.continous_columns,
                            categorical=self.categorical_columns,
                            immutables=self.nonactionable_columns,
                            target=self.target_feature_name)

        self.model = TFModelAdult(model=explained_model, data=self.data_catalog, columns_ohe_order=columns_order_ohe)


        # init explainers
        self.gs_explainer = GrowingSpheres(self.model)

        ar_hyperparams = {
            'fs_size': 100
        }
        self.ar_explainer = ActionableRecourse(self.model, ar_hyperparams)

    def generate_counterfactuals(self, query_instance: pd.DataFrame, 
        growing_spheres_restarts: int = 15, face_restarts: int = 10,
        ) -> pd.DataFrame:
        '''
        Growing spheres is fast so can have more restarts than face
        '''

        to_concat = list()
        explainers_list = list()
        query_instance_ohe_norm = self.data_catalog.transform(query_instance)

        for _ in tqdm(range(growing_spheres_restarts), desc='Growing Spheres generating'):
            gs_cf = self.gs_explainer.get_counterfactuals(query_instance_ohe_norm)
            to_concat.append(gs_cf)
            explainers_list.append('growing-spheres')

        # AR only once because it gets the same cf
        ar_cf = self.ar_explainer.get_counterfactuals(query_instance_ohe_norm)
        to_concat.append(ar_cf)
        explainers_list.append('actionable-recourse')


        for i in tqdm(range(face_restarts), desc='FACE generating'):
            face_hyperparams = {
                'mode': 'knn',
                'fraction': 0.05,
            }
            self.face_explainer = Face(self.model, face_hyperparams)
            face_cf = self.face_explainer.get_counterfactuals(query_instance_ohe_norm)
            to_concat.append(face_cf)
            explainers_list.append('face')


        concatenated = pd.concat(to_concat, ignore_index=True).reset_index(drop=True)
        #concatenated['explainer'] = explainers_list
        return concatenated, explainers_list

        

        