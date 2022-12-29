from typing import Dict
from carla import MLModel
import tensorflow as tf
import pandas as pd
import json

class TFModelAdult(MLModel):

    def __init__(self, data) -> None:
        super().__init__(data)
        self.__load_constraints()
        self._mymodel = self.__load_model()
        

    # List of the feature order the ml model was trained on
    @property
    def feature_input_order(self):
        return self.constraints['features_order']

    # The ML framework the model was trained on
    @property
    def backend(self):
        return "tensorflow"

    # The black-box model object
    @property
    def raw_model(self):
        return self._mymodel

    # The predict function outputs
    # the continuous prediction of the model
    def predict(self, x):

        if isinstance(x, pd.DataFrame):
            x = x[self.feature_input_order]

        return self._mymodel.predict(x)

    # The predict_proba method outputs
    # the prediction as class probabilities
    def predict_proba(self, x):

        if isinstance(x, pd.DataFrame):
            x = x[self.feature_input_order]

        return self._mymodel.predict_proba(x)

    def __load_model(self, filepath: str = '../models/adult_NN.h5') -> tf.keras.Model:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input((self.constraints['features_count'],)))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(2, activation='softmax'))

        model.load_weights(filepath)

        return model

    def __load_constraints(self, filepath: str = '../data/adult_constraints.json') -> Dict:
        with open('../data/adult_constraints.json') as f:
            self.constraints = json.load(f)
        return self.constraints