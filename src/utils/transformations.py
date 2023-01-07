import pandas as pd
import numpy as np

def transform_to_sparse(
    _df: pd.DataFrame, 
    original_df: pd.DataFrame,
    categorical_features: list,
    continuous_features: list,
    ) -> pd.DataFrame:
    """
    Transforms a dataframe into a sparse dataframe.

    Args:
        df (pd.DataFrame): Dataframe to be transformed.
        original_df (pd.DataFrame): Original dataframe.
        categorical_features (list): List of categorical features.
        continuous_features (list): List of continuous features.

    Returns:
        pd.DataFrame: Sparse dataframe.
    """
    to_concat = []

    # Add df to original df to get all possible values for categorical features
    df = pd.concat([_df, original_df], axis=0, ignore_index=True) 


    for feature in continuous_features:
        to_concat.append(df[feature])

    for feature in categorical_features:
        to_concat.append(pd.get_dummies(df[feature], prefix=feature))

    sparse_df = pd.concat(to_concat, axis=1)

    # Get only the rows that belong to the _df
    sparse_df = sparse_df.iloc[:len(_df)]
    
    return sparse_df

def inverse_transform_to_sparse(
    sparse_df: pd.DataFrame,
    original_df: pd.DataFrame,
    categorical_features: list,
    continuous_features: list,
    ) -> pd.DataFrame:
    """
    Inverse transforms a sparse dataframe into a dataframe.

    Args:
        sparse_df (pd.DataFrame): Sparse dataframe to be transformed.
        original_df (pd.DataFrame): Original dataframe.
        categorical_features (list): List of categorical features.
        continuous_features (list): List of continuous features.
    
    Returns:
        pd.DataFrame: Dataframe.
    """
    dense_df = pd.DataFrame(columns=continuous_features + categorical_features)

    for feature in continuous_features:
        dense_df[feature] = sparse_df[feature]

    for feature in categorical_features:
        split_columns = [col for col in sparse_df.columns if feature in col]
        # Get columns that belong to the feature. They should be oneHot encoded. If they are not, they are not categorical, and error should be thrown.
        dense_df[feature] = sparse_df[split_columns].idxmax(axis=1).str.replace(feature + '_', '')

    # Change datatypes to original datatypes
    for feature in dense_df.columns:
        dense_df[feature] = dense_df[feature].astype(original_df[feature].dtype)
    
    return dense_df

def min_max_normalization(
    _df: pd.DataFrame,
    original_df: pd.DataFrame,
    continuous_features: list,
    ) -> pd.DataFrame:
    """
    Normalizes a dataframe using min-max normalization.

    Args:
        df (pd.DataFrame): Dataframe to be normalized.
        original_df (pd.DataFrame): Original dataframe.
        continuous_features (list): List of continuous features.
    
    Returns:
        pd.DataFrame: Normalized dataframe.
    """
    df = _df.copy()
    for feature in continuous_features:
        df[feature] = (df[feature] - original_df[feature].min()) / (original_df[feature].max() - original_df[feature].min())
    
    return df

def inverse_min_max_normalization(
    _df: pd.DataFrame,
    original_df: pd.DataFrame,
    continuous_features: list,
    ) -> pd.DataFrame:
    """
    Inverse normalizes a dataframe using min-max normalization.

    Args:
        df (pd.DataFrame): Dataframe to be inverse normalized.
        original_df (pd.DataFrame): Original dataframe.
        continuous_features (list): List of continuous features.

    Returns:
        pd.DataFrame: Inverse normalized dataframe.
    """   
    df = _df.copy()
    for feature in continuous_features:
        df[feature] = df[feature] * (original_df[feature].max() - original_df[feature].min()) + original_df[feature].min()

    return df


if __name__ == '__main__':
    # sparse = transform_to_sparse(
    #     df=train_dataset,
    #     categorical_features=constr['categorical_features_nonsplit'],
    #     continuous_features=constr['continuous_features_nonsplit']
    # )

    # inverse_sparse = inverse_transform_to_sparse(
    #     sparse_df=sparse,
    #     categorical_features=constr['categorical_features_nonsplit'],
    #     continuous_features=constr['continuous_features_nonsplit']
    # )
    pass