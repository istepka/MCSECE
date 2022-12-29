from carla import Data

def load_dataset_from_disk(filepath: str = '../data/adult_cleaned.csv') -> 

# Custom data set implementations need to inherit from the Data interface
class MyOwnDataSet(Data):
    def __init__(self):
        # The data set can e.g. be loaded in the constructor
        self._dataset = load_dataset_from_disk()

    # List of all categorical features
    @property
    def categorical(self):
        return [...]

    # List of all continuous features
    @property
    def continuous(self):
        return [...]

    # List of all immutable features which
    # should not be changed by the recourse method
    @property
    def immutables(self):
        return [...]

    # Feature name of the target column
    @property
    def target(self):
        return "label"

    # The full dataset
    @property
    def df(self):
        return self._dataset

    # The training split of the dataset
    @property
    def df_train(self):
        return self._dataset_train

    # The test split of the dataset
    @property
    def df_test(self):
         return self._dataset_test

    # Data transformation, for example normalization of continuous features
    # and encoding of categorical features
    def transform(self, df):
         return transformed_df

    # Inverts transform operation
    def inverse_transform(self, df):
         return original_df