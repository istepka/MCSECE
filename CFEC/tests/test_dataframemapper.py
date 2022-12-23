from cfec.preprocessing import DataFrameMapper
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def _transform_to_numpy(df):
    df_numpy = df.copy()

    df_numpy = df_numpy.drop(columns=['b'])
    df_numpy = StandardScaler().fit_transform(df_numpy)
    return np.hstack([df_numpy, np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3)])


def _default_dataframe():
    return pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                        columns=['a', 'b', 'c'])


def test_dataframemapper():
    df2 = _default_dataframe()
    mapper = DataFrameMapper(nominal_columns=['b'])

    mapper.fit(x=df2)
    mapped = mapper.transform(df2)

    df_numpy = _transform_to_numpy(df2)

    assert np.array_equal(df_numpy, mapped)


def test_dataframemapper_fit_transform():
    df2 = _default_dataframe()
    mapper = DataFrameMapper(nominal_columns=['b'])

    mapped = mapper.fit_transform(df2)

    df_numpy = _transform_to_numpy(df2)

    np.testing.assert_array_equal(df_numpy, mapped)


def test_all_continuous():
    df = pd.DataFrame(data=np.random.rand(3, 3))
    df_transformed = StandardScaler().fit_transform(df)

    mapper = DataFrameMapper(nominal_columns=[])

    mapper_df_transformed = mapper.fit_transform(df)

    np.testing.assert_array_equal(df_transformed, mapper_df_transformed)

    pd.testing.assert_frame_equal(df, mapper.inverse_transform(mapper_df_transformed))


def test_all_nominal():
    df = pd.DataFrame(data=np.random.randint(0, 100, (3, 3)))
    df_one_hot = OneHotEncoder(sparse=False).fit_transform(df)

    mapper = DataFrameMapper(nominal_columns=[0, 1, 2])

    mapper_df_transformed = mapper.fit_transform(df)

    np.testing.assert_array_equal(df_one_hot, mapper_df_transformed)

    pd.testing.assert_frame_equal(df, mapper.inverse_transform(mapper_df_transformed))
