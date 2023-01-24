import dice_ml
import pandas as pd
import numpy as np
import json 
import pickle
import sklearn
from sklearn.ensemble import RandomForestClassifier
from dice_ml.model_interfaces.base_model import BaseModel
from dice_ml.model import ModelTypes
import tensorflow as tf


class SklearnModelDice(BaseModel):

    def __init__(self, model=None, model_path='', backend='', func=None, kw_args=None):
        super().__init__(model, model_path, backend, func, kw_args)
        self.model_type = ModelTypes.Classifier
        
    def load_model(self):
        if self.model_path != '':
            with open(self.model_path, 'rb') as filehandle:
                self.model = pickle.load(filehandle)

    def get_output(self, input_instance, model_score=True):
        """returns prediction probabilities for a classifier and the predicted output for a regressor.
        :returns: an array of output scores for a classifier, and a singleton
        array of predicted value for a regressor.
        """
        input_instance = self.transformer.transform(input_instance)
        if model_score:
            if self.model_type == dice_ml.model.ModelTypes.Classifier:
                # THIS IS DIFFERENT FROM ORIGINAL IMPLEMENTATION BECAUSE OTHERWISE 
                # THERE WAS AN ERROR BECAUSE RETURNED TYPE WAS LIST INSTEAD OF NP.ARRAY
                output = self.model.predict_proba(input_instance)[0]
                if type(output) != np.ndarray:
                    output = np.array(output)
                return output
            else:
                return self.model.predict(input_instance)
        else:
            return self.model.predict(input_instance)

class DiceModel: 

    def __init__(self, train_dataset: pd.DataFrame, 
        continuous_features: list,
        categorical_features: list,
        target: str,
        backend: str,
        model: tf.keras.Model | RandomForestClassifier,
        func: str = 'ohe-min-max',
        ) -> None:

        self.train_dataset = train_dataset
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.target = target
        self.backend = backend
        self.func = func

        self.Data =  dice_ml.Data(
            dataframe=self.train_dataset, 
            continuous_features=self.continuous_features, 
            categorical_features=self.categorical_features,
            outcome_name=self.target
            )

        if isinstance(model, tf.keras.Model):
            self.Model = dice_ml.Model(
                #model_path=self.model_path, 
                backend='TF2', 
                model=model,
                func=self.func
                )
        else:
            self.Model = SklearnModelDice(
                backend='sklearn', 
                model=model,
                func=self.func          
            )

        self.Dice = dice_ml.Dice(
            self.Data,
            self.Model,
            method='random'
            )

    def generate_counterfactuals(self, query_instance: pd.DataFrame, 
        total_CFs: int,
        desired_class: str,
        features_to_vary: str,
        permitted_range: dict,
        ) -> pd.DataFrame:

        explanation = self.Dice.generate_counterfactuals(
            query_instances=query_instance,
            total_CFs=total_CFs,
            desired_class=desired_class,
            features_to_vary=features_to_vary,
            permitted_range=permitted_range,
            )

        cfs = json.loads(explanation.cf_examples_list[0].to_json(dice_ml.constants._SchemaVersions.V2))
        dataframe = pd.DataFrame(cfs['final_cfs_list'], columns=self.train_dataset.columns)

        return dataframe

    def transform_to_normalized(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return self.Data.get_ohe_min_max_normalized_data(dataframe)



#     def get_gradient(self):
#         raise NotImplementedError

#     def get_num_output_nodes(self, inp_size):
#         temp_input = np.transpose(np.array([np.random.uniform(0, 1) for i in range(inp_size)]).reshape(-1, 1))
#         return self.get_output(temp_input).shape[1]

#     def get_num_output_nodes2(self, input_instance):
#         if self.model_type == ModelTypes.Regressor:
#             raise SystemException('Number of output nodes not supported for regression')
#         return self.get_output(input_instance).shape[1]