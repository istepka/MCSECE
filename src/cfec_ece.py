from modules.CFEC.cfec.explainers import Fimap, ECE
import pandas as pd


class CfecEceModel:

    def __init__(self, train_data, model_predictions, constraints_dictionary) -> None:

        fimaps = []
        fimap_hyperparameters = [
            (0.1, 0.001, 0.01),
            (0.1, 0.05, 0.5),
            # (0.2, 0.01, 0.1),
            # (0.2, 0.08, 0.8),
            # (0.5, 0.001, 0.01)
        ]

        for tau, l1, l2 in fimap_hyperparameters:
            fimap = Fimap(tau, l1, l2, use_mapper=True)
            fimap.fit(train_data, model_predictions)
            fimaps.append(fimap)

        self.ece = ECE(3, columns=list(train_data.columns), bces=fimaps, dist=2, h=5, lambda_=0.001, n_jobs=1)

    def generate_counterfactuals(self, query_instance: pd.Series) -> pd.DataFrame:
        cfs = self.ece.generate(query_instance)
        return cfs

    def __transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # TODO
        pass 

    def __inverse_transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # TODO
        pass
            
