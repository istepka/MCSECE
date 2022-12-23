import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from cfec.constraints import Freeze, OneHot, ValueMonotonicity


class GermanData:
    def __init__(self, input_file, labels_file, test_frac=0.2):
        self.input = pd.read_csv(input_file, index_col=0, dtype=np.float64)
        self.labels = pd.read_csv(labels_file, index_col=0, dtype=np.int32)
        self.constraints = [OneHot("account_status", 7, 10), OneHot("credit_history", 11, 15),
                            OneHot("purpose", 16, 25), OneHot("savings", 26, 30), OneHot("sex_status", 31, 34),
                            OneHot("debtors", 35, 37), OneHot("property", 38, 41),
                            OneHot("other_installment_plans", 42, 44), OneHot("housing", 45, 47), OneHot("job", 48, 51),
                            OneHot("phone", 52, 53), OneHot("foreign", 54, 55), OneHot("employment", 56, 60)]

        self.additional_constraints = [Freeze(['credit']), ValueMonotonicity(['age'], "increasing")]
        self.index = 0

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.input, self.labels, test_size=test_frac)

        self.num_input_columns = len(self.input.columns)
        self.input_columns = self.input.columns
        self.num_label_columns = len(self.labels.columns)
        self.label_columns = self.labels.columns

        self._scaler = MinMaxScaler()
        self._scaler.fit(self.X_train.to_numpy(dtype=np.float64))
        self.X_train = pd.DataFrame(self._scaler.transform(self.X_train.to_numpy(dtype=np.float64)), index=self.X_train.index,
                                    columns=self.X_train.columns)

        if self.X_test.shape[0] > 0:
            self.X_test = pd.DataFrame(self._scaler.transform(self.X_test.to_numpy(dtype=np.float64)), index=self.X_test.index,
                                       columns=self.X_test.columns)

    def unscale(self, data):
        if type(data) is pd.DataFrame:
            return pd.DataFrame(self._scaler.inverse_transform(data.to_numpy()), index=data.index, columns=data.columns)

        elif type(data) is pd.Series:
            return pd.Series(self._scaler.inverse_transform([data.to_numpy()])[0].transpose(), index=data.index)
        else:
            return self._scaler.inverse_transform([data])[0]

    def scale(self, data):
        if type(data) is pd.DataFrame:
            return pd.DataFrame(self._scaler.transform(data.to_numpy()), index=data.index, columns=data.columns)
        elif type(data) is pd.Series:
            return pd.Series(self._scaler.transform([data.to_numpy()])[0].transpose(), index=data.index)
        else:
            return self._scaler.transform([data])[0]
