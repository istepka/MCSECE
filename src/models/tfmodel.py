from typing import Dict, List
from carla import MLModel
import tensorflow as tf
import pandas as pd
import json

class TFModelAdult(MLModel):

    def __init__(self, model: tf.keras.Model, data: pd.DataFrame, columns_ohe_order: List[str]) -> None:
        super().__init__(data)
        self._mymodel = model#self.__load_model()

        self.columns_order = columns_ohe_order
    
    def __call__(self, data):
        return self._mymodel(data)

    # List of the feature order the ml model was trained on
    @property
    def feature_input_order(self):
        return self.columns_order

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
            x = x[self.feature_input_order].to_numpy()
            print(x)
            

        if isinstance(x, tf.Variable):
            #print(f'X, type: {type} data: {x}')
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                x = x.eval(session=sess)
            #print(f'X, type: {type} data: {x}')

        return self._mymodel.predict(x)

        
    # @tf.function(experimental_relax_shapes=True)
    # def predictTensor(self, x):
    #     self._mymodel.predict(x, steps=1)

    # The predict_proba method outputs
    # the prediction as class probabilities
    def predict_proba(self, x):

        if isinstance(x, pd.DataFrame):
            x = x[self.feature_input_order]

        return self._mymodel.predict(x)

    