import dice_ml
import pandas as pd
import json 


class DiceModel: 

    def __init__(self, train_dataset: pd.DataFrame, 
        continuous_features: list,
        categorical_features: list,
        target: str,
        model_path: str,
        backend: str,
        func: str,
        ) -> None:

        self.train_dataset = train_dataset
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.target = target
        self.model_path = model_path
        self.backend = backend
        self.func = func

        self.Data =  dice_ml.Data(
            dataframe=self.train_dataset, 
            continuous_features=self.continuous_features, 
            categorical_features=self.categorical_features,
            outcome_name=self.target
            )

        self.Model = dice_ml.Model(
            model_path=self.model_path, 
            backend=self.backend, 
            func=self.func
            )

        self.Dice = dice_ml.Dice(
            self.Data,
            self.Model,
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