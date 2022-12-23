from typing import Optional, List

import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from cfec.constraints import Freeze, OneHot, ValueMonotonicity, ValueNominal


class AdultData:
    def __init__(self, dataset_file: str, columns_to_drop: Optional[List[str]] = None, test_frac=0.2, random_state=42):

        df = pd.read_csv(dataset_file)
        df = df.sample(frac=1)
        df.replace('?', np.NaN, inplace=True)
        df.dropna(inplace=True)

        self.target_column = 'income'

        categorical_columns = ['workclass', 'education', 'marital.status', 'occupation',
                               'relationship', 'race', 'sex', 'native.country']
        freeze_columns = ['race', 'sex', 'native.country']

        

        if columns_to_drop is None:
            columns_to_drop = ['fnlwgt', 'education.num', 'education', 'relationship', 'native.country', 'capital.gain',
                               'capital.loss']
        df.drop(columns_to_drop, axis=1, inplace=True)
        self.categorical_columns = [column for column in categorical_columns if column not in columns_to_drop]
        self.freeze_columns = [column for column in freeze_columns if column not in columns_to_drop]

        self.raw_X_train, self.raw_X_test, self.raw_y_train, self.raw_y_test = \
            train_test_split(df.drop(columns=[self.target_column]), df[self.target_column], test_size=test_frac,
                             random_state=random_state)

        self.label_encoders = []
        for feature in self.categorical_columns + [self.target_column]:
            lb = LabelEncoder()
            df[feature] = lb.fit_transform(df[feature])
            self.label_encoders.append(lb)

        self.encoded_raw_X_train, self.encoded_raw_X_test, self.encoded_raw_y_train, self.encoded_raw_y_test = \
            train_test_split(df.drop(columns=[self.target_column]), df[self.target_column], test_size=test_frac,
                             random_state=random_state)

        self.standard_scalers = []
        for feature in df.columns:
            if feature not in self.categorical_columns and feature != self.target_column:
                sc = MinMaxScaler()
                df[feature] = sc.fit_transform(df[feature].values.reshape(-1, 1))
                self.standard_scalers.append(sc)

        for feature in self.categorical_columns:
            one_hot = pd.get_dummies(df[feature])
            one_hot = one_hot.add_prefix(f"{feature}-")
            df.drop(feature, axis = 1, inplace=True)
            df = df.join(one_hot)

        df_inputs = df.drop(columns=[self.target_column])
        df_labels = df[self.target_column]
        
        one_hot_constraints = [
            OneHot('workclass', 2, 8),
            OneHot('martial.status', 9, 15),
            OneHot('occupation', 16, 29),
            OneHot('race', 30, 34),
            OneHot('sex', 35, 36),
        ]

        self.constraints = [
            #ValueNominal(columns=self.categorical_columns), 
            *one_hot_constraints,
        ]

        self.additional_constraints = [
            Freeze(columns=self.freeze_columns)
            ]
        if 'age' not in columns_to_drop:
            self.additional_constraints.append(ValueMonotonicity(['age'], 'increasing'))

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(df_inputs, df_labels, test_size=test_frac, random_state=random_state)

        self.X_train, self.y_train = RandomUnderSampler().fit_resample(self.X_train, self.y_train)
        self.raw_X_train, self.raw_y_train = RandomUnderSampler().fit_resample(self.raw_X_train, self.raw_y_train)
        self.encoded_raw_X_train, self.encoded_raw_y_train = RandomUnderSampler().fit_resample(self.encoded_raw_X_train,
                                                                                               self.encoded_raw_y_train)

        self.y_train_binarized = pd.get_dummies(self.y_train)
        self.y_test_binarized = pd.get_dummies(self.y_test)

    def inverse_transform(self, X: pd.DataFrame, y=None):
        for label_encoder, column in self.categorical_columns:
            X[column] = label_encoder.inverse_transform(X[column])

        if y is not None:
            if isinstance(y, pd.DataFrame):
                y[self.target_column] = self.label_encoders[-1].inverse_transform(y[self.target_column])
            else:
                y = pd.DataFrame(data=self.label_encoders[-1].inverse_transform(y), columns=[self.target_column])
            return X, y
        else:
            return X
